#include <Arduino.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "experiment_models.h"

namespace {
constexpr size_t NFFT=2048, HOP=1024, NBINS=1025, K=48, BLOCK=512;
constexpr float PI_F=3.14159265358979323846f;
constexpr uint16_t STEPS[]={8,16,32,64,128,256,512,1024,2048,4096};
constexpr size_t N_STEPS=sizeof(STEPS)/sizeof(STEPS[0]);
constexpr size_t N_PROFILES=3+N_STEPS+1;
const char *PROFILE_NAMES[N_PROFILES]={"legacy_nnls","m64_nnls","m128_nnls","m64_fista8","m64_fista16","m64_fista32","m64_fista64","m64_fista128","m64_fista256","m64_fista512","m64_fista1024","m64_fista2048","m64_fista4096","m128_fista512"};
struct ProfileResult { char name[24]; float h[K]; float event; uint32_t solver_us; };
struct PeakCandidate { uint32_t index; float height; float prominence; };

float windowFn[NFFT], realPart[NFFT], imagPart[NFFT], frame[NFFT];
float magnitude[NBINS], oldFeatures[NBINS], mel64[64], mel128[128];
float gramOld[K*K], gram64[K*K], gram128[K*K];
float rhs[K], cdH[K], fistaX[K], fistaY[K], fistaNext[K];
uint8_t rx[BLOCK*sizeof(float)];
ProfileResult outputs[3+N_STEPS+1];
float *eventScores=nullptr; size_t scoreCapacity=0;
float peakProminence[N_PROFILES]{}; uint16_t peakDistance[N_PROFILES]{}; bool peakConfigured[N_PROFILES]{};
size_t frameFill=0, hopFill=0, frameCount=0;
uint64_t expectedSamples=0, samplesSeen=0;
float fileMean=0; bool streamActive=false, ready=false;
char fileId[64]="-", dataSplit[24]="unknown"; int truthCount=-1;
uint64_t totalFrontendUs=0, minFrameTotalUs=UINT64_MAX, maxFrameTotalUs=0;
uint32_t minHeapAtStart=UINT32_MAX, minPsramAtStart=UINT32_MAX;

void fft(){
  for(size_t i=1,j=0;i<NFFT;i++){size_t bit=NFFT>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;
    if(i<j){std::swap(realPart[i],realPart[j]);std::swap(imagPart[i],imagPart[j]);}}
  for(size_t len=2;len<=NFFT;len<<=1){float a=-2*PI_F/(float)len,wrs=cosf(a),wis=sinf(a);
    for(size_t base=0;base<NFFT;base+=len){float wr=1,wi=0;for(size_t j=0;j<len/2;j++){
      size_t u=base+j,v=u+len/2;float vr=realPart[v]*wr-imagPart[v]*wi,vi=realPart[v]*wi+imagPart[v]*wr;
      float ur=realPart[u],ui=imagPart[u];realPart[u]=ur+vr;imagPart[u]=ui+vi;realPart[v]=ur-vr;imagPart[v]=ui-vi;
      float t=wr*wrs-wi*wis;wi=wr*wis+wi*wrs;wr=t;}}}
}

void makeFeatures(){
  for(size_t n=0;n<NFFT;n++){realPart[n]=frame[n]*windowFn[n];imagPart[n]=0;}
  fft();
  for(size_t f=0;f<NBINS;f++){magnitude[f]=sqrtf(realPart[f]*realPart[f]+imagPart[f]*imagPart[f]);oldFeatures[f]=log1pf(magnitude[f]);}
  for(size_t m=0;m<64;m++){float sum=0;for(uint16_t q=AVS_MEL64_OFFSETS[m];q<AVS_MEL64_OFFSETS[m+1];q++)sum+=AVS_MEL64_WEIGHTS[q]*magnitude[AVS_MEL64_BIN_INDEX[q]];mel64[m]=log1pf(sum/AVS_MEL64_REFERENCE);}
  for(size_t m=0;m<128;m++){float sum=0;for(uint16_t q=AVS_MEL128_OFFSETS[m];q<AVS_MEL128_OFFSETS[m+1];q++)sum+=AVS_MEL128_WEIGHTS[q]*magnitude[AVS_MEL128_BIN_INDEX[q]];mel128[m]=log1pf(sum/AVS_MEL128_REFERENCE);}
}

void buildGram(const float *w,size_t rows,float *gram){
  for(size_t i=0;i<K;i++)for(size_t j=0;j<K;j++){float sum=0;for(size_t f=0;f<rows;f++)sum+=w[f*K+i]*w[f*K+j];gram[i*K+j]=sum;}
}
void calcRhs(const float *w,const float *v,size_t rows){
  for(size_t j=0;j<K;j++){float sum=0;for(size_t f=0;f<rows;f++)sum+=w[f*K+j]*v[f];rhs[j]=sum;}
}
float eventSum(const float *h){float sum=0;for(size_t j=0;j<24;j++)sum+=h[j];return sum;}

size_t outputCount=0;uint64_t currentComputeUs=0;
void saveActivation(const char *profile,const float *h,uint32_t solverUs){
  if(outputCount>=sizeof(outputs)/sizeof(outputs[0]))return;
  ProfileResult &out=outputs[outputCount++];strlcpy(out.name,profile,sizeof(out.name));
  memcpy(out.h,h,sizeof(out.h));out.event=eventSum(h);out.solver_us=solverUs;
}
void emitMel(const char *name,const float *v,size_t rows){
  Serial.printf("%s,%s,%u",name,fileId,(unsigned)frameCount);
  for(size_t i=0;i<rows;i++)Serial.printf(",%.7g",(double)v[i]);
  Serial.println();
}

void solveCd(const float *w,const float *v,size_t rows,const float *gram,const char *name,uint32_t frontUs){
  const uint32_t t0=(uint32_t)esp_timer_get_time();calcRhs(w,v,rows);memset(cdH,0,sizeof(cdH));
  for(unsigned sweep=0;sweep<80;sweep++)for(size_t j=0;j<K;j++){
    float residual=rhs[j];for(size_t k=0;k<K;k++)if(k!=j)residual-=gram[j*K+k]*cdH[k];
    const float diagonal=gram[j*K+j];cdH[j]=diagonal>1e-20f?fmaxf(0,residual/diagonal):0;
  }
  const uint32_t elapsed=(uint32_t)esp_timer_get_time()-t0;currentComputeUs+=elapsed;saveActivation(name,cdH,elapsed);
}

void solveFista64(const uint32_t frontUs){
  const uint32_t start=(uint32_t)esp_timer_get_time();calcRhs(AVS_MEL64_W,mel64,64);memset(fistaX,0,sizeof(fistaX));memset(fistaY,0,sizeof(fistaY));
  float lipschitz=0;for(size_t i=0;i<K;i++){float row=0;for(size_t j=0;j<K;j++)row+=gram64[i*K+j];lipschitz=fmaxf(lipschitz,row);}
  if(!(lipschitz>0))return;
  float t=1;size_t target=0;
  for(uint32_t iter=1;iter<=STEPS[N_STEPS-1];iter++){
    float tn=.5f*(1+sqrtf(1+4*t*t));
    for(size_t j=0;j<K;j++){float gy=0;for(size_t k=0;k<K;k++)gy+=gram64[j*K+k]*fistaY[k];fistaNext[j]=fmaxf(0,fistaY[j]-(gy-rhs[j])/lipschitz);}
    for(size_t j=0;j<K;j++)fistaY[j]=fistaNext[j]+((t-1)/tn)*(fistaNext[j]-fistaX[j]);
    memcpy(fistaX,fistaNext,sizeof(fistaX));t=tn;
    if(target<N_STEPS&&iter==STEPS[target]){char name[24];snprintf(name,sizeof(name),"m64_fista%u",(unsigned)iter);
      saveActivation(name,fistaX,(uint32_t)esp_timer_get_time()-start);target++;}
  }
  currentComputeUs+=(uint32_t)esp_timer_get_time()-start;
}

void solveFista128(const uint32_t frontUs){
  const uint32_t start=(uint32_t)esp_timer_get_time();calcRhs(AVS_MEL128_W,mel128,128);memset(fistaX,0,sizeof(fistaX));memset(fistaY,0,sizeof(fistaY));
  float lipschitz=0;for(size_t i=0;i<K;i++){float row=0;for(size_t j=0;j<K;j++)row+=gram128[i*K+j];lipschitz=fmaxf(lipschitz,row);}
  if(!(lipschitz>0))return;float t=1;
  for(uint32_t iter=1;iter<=512;iter++){float tn=.5f*(1+sqrtf(1+4*t*t));
    for(size_t j=0;j<K;j++){float gy=0;for(size_t k=0;k<K;k++)gy+=gram128[j*K+k]*fistaY[k];fistaNext[j]=fmaxf(0,fistaY[j]-(gy-rhs[j])/lipschitz);}
    for(size_t j=0;j<K;j++)fistaY[j]=fistaNext[j]+((t-1)/tn)*(fistaNext[j]-fistaX[j]);memcpy(fistaX,fistaNext,sizeof(fistaX));t=tn;}
  const uint32_t elapsed=(uint32_t)esp_timer_get_time()-start;currentComputeUs+=elapsed;saveActivation("m128_fista512",fistaX,elapsed);
}

void processFrame(){
  const uint32_t frameStart=(uint32_t)esp_timer_get_time();makeFeatures();const uint32_t frontUs=(uint32_t)esp_timer_get_time()-frameStart;
  totalFrontendUs+=frontUs;outputCount=0;currentComputeUs=frontUs;
  solveCd(AVS_OLD_W,oldFeatures,NBINS,gramOld,"legacy_nnls",frontUs);
  solveCd(AVS_MEL64_W,mel64,64,gram64,"m64_nnls",frontUs);
  solveCd(AVS_MEL128_W,mel128,128,gram128,"m128_nnls",frontUs);
  solveFista64(frontUs);solveFista128(frontUs);
  minFrameTotalUs=min(minFrameTotalUs,currentComputeUs);maxFrameTotalUs=max(maxFrameTotalUs,currentComputeUs);
  if(eventScores&&frameCount<scoreCapacity)for(size_t i=0;i<outputCount;i++)eventScores[frameCount*N_PROFILES+i]=outputs[i].event;
  const uint32_t outputStart=(uint32_t)esp_timer_get_time();
  emitMel("MEL64",mel64,64);emitMel("MEL128",mel128,128);
  for(size_t i=0;i<outputCount;i++){const ProfileResult &out=outputs[i];
    Serial.printf("ACT,%s,%s,%u,%s,%u,%u,%.8g",dataSplit,fileId,(unsigned)frameCount,out.name,
                  (unsigned)frontUs,(unsigned)out.solver_us,(double)out.event);
    for(size_t j=0;j<K;j++)Serial.printf(",%.7g",(double)out.h[j]);Serial.println();}
  Serial.flush();const uint32_t telemetryUs=(uint32_t)esp_timer_get_time()-outputStart;
  Serial.printf("FRAME_WALL,%s,%u,compute_us=%llu,telemetry_us=%u,wall_us=%llu\n",fileId,(unsigned)frameCount,
    (unsigned long long)currentComputeUs,(unsigned)telemetryUs,(unsigned long long)(currentComputeUs+telemetryUs));
  frameCount++;
}

void addSample(float sample){sample-=fileMean;
  if(frameFill<NFFT){frame[frameFill++]=sample;if(frameFill==NFFT)processFrame();return;}
  frame[HOP+hopFill++]=sample;if(hopFill==HOP){memmove(frame,frame+HOP,(NFFT-HOP)*sizeof(float));hopFill=0;processFrame();}
}

void printMemory(){
  const uint32_t freeHeap=ESP.getFreeHeap(),minHeap=ESP.getMinFreeHeap();
  const uint32_t psramTotal=ESP.getPsramSize(),psramFree=ESP.getFreePsram();
  const uint32_t sketch=ESP.getSketchSize(),sketchFree=ESP.getFreeSketchSpace();
  const uint32_t stackWords=uxTaskGetStackHighWaterMark(nullptr);
  Serial.printf("MEM,free_heap=%u,min_free_heap=%u,psram_total=%u,psram_free=%u,flash_app=%u,flash_free=%u,stack_watermark_words=%u\n",
    (unsigned)freeHeap,(unsigned)minHeap,(unsigned)psramTotal,(unsigned)psramFree,(unsigned)sketch,(unsigned)sketchFree,(unsigned)stackWords);
}
void resetFile(){if(eventScores){free(eventScores);eventScores=nullptr;}scoreCapacity=0;frameFill=hopFill=frameCount=0;samplesSeen=0;totalFrontendUs=0;minFrameTotalUs=UINT64_MAX;maxFrameTotalUs=0;minHeapAtStart=ESP.getFreeHeap();minPsramAtStart=ESP.getFreePsram();}
int profileIndex(const char *name){for(size_t i=0;i<N_PROFILES;i++)if(strcmp(name,PROFILE_NAMES[i])==0)return (int)i;return -1;}
size_t detectPeaks(const float *env,size_t length,float prominence,uint16_t distance,uint32_t *picked){
  if(length<3||!env)return 0;
  PeakCandidate *candidates=(PeakCandidate*)heap_caps_malloc(length*sizeof(PeakCandidate),MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
  if(!candidates)candidates=(PeakCandidate*)malloc(length*sizeof(PeakCandidate));
  if(!candidates)return 0;
  size_t count=0;
  for(size_t i=1;i+1<length;){
    if(env[i]>env[i-1]){size_t end=i;while(end+1<length&&env[end+1]==env[i])end++;
      if(end+1<length&&env[end]>env[end+1]){const size_t peak=(i+end)/2;const float height=env[peak];
        float leftMin=height,rightMin=height;size_t j=peak;
        while(j>0&&env[j-1]<=height){--j;leftMin=fminf(leftMin,env[j]);}
        j=peak;while(j+1<length&&env[j+1]<=height){++j;rightMin=fminf(rightMin,env[j]);}
        const float p=height-fmaxf(leftMin,rightMin);
        if(p>=prominence)candidates[count++]={static_cast<uint32_t>(peak),height,p};}
      i=end+1;
    }else ++i;
  }
  std::sort(candidates,candidates+count,[](const PeakCandidate&a,const PeakCandidate&b){return a.height>b.height;});
  size_t pickedCount=0;const uint16_t minDistance=distance?distance:1;
  for(size_t c=0;c<count;c++){bool keep=true;for(size_t p=0;p<pickedCount;p++){
      const uint32_t delta=candidates[c].index>picked[p]?candidates[c].index-picked[p]:picked[p]-candidates[c].index;
      if(delta<minDistance){keep=false;break;}}
    if(keep)picked[pickedCount++]=candidates[c].index;
  }
  free(candidates);std::sort(picked,picked+pickedCount);return pickedCount;
}
void printPeakResults(){
  if(!eventScores)return;uint32_t *picked=(uint32_t*)heap_caps_malloc(scoreCapacity*sizeof(uint32_t),MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
  if(!picked)picked=(uint32_t*)malloc(scoreCapacity*sizeof(uint32_t));if(!picked){Serial.println("ERROR,PEAK_BUFFER");return;}
  for(size_t p=0;p<N_PROFILES;p++)if(peakConfigured[p]){
    float *profileEnv=(float*)heap_caps_malloc(scoreCapacity*sizeof(float),MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
    if(!profileEnv)profileEnv=(float*)malloc(scoreCapacity*sizeof(float));if(!profileEnv)continue;
    for(size_t f=0;f<frameCount;f++)profileEnv[f]=eventScores[f*N_PROFILES+p];
    const size_t n=detectPeaks(profileEnv,frameCount,peakProminence[p],peakDistance[p],picked);
    Serial.printf("PEAK,%s,%s,%s,%.8g,%u,%u",dataSplit,fileId,PROFILE_NAMES[p],(double)peakProminence[p],(unsigned)peakDistance[p],(unsigned)n);
    for(size_t i=0;i<n;i++)Serial.printf(",%u",(unsigned)picked[i]);Serial.println();free(profileEnv);
  }
  free(picked);
}
void finishFile(){
  if(!streamActive||samplesSeen!=expectedSamples){Serial.printf("ERROR,END_COUNT,%s,%llu,%llu\n",fileId,(unsigned long long)expectedSamples,(unsigned long long)samplesSeen);streamActive=false;return;}
  streamActive=false;const uint64_t expectedFrames=expectedSamples<NFFT?0:1+(expectedSamples-NFFT)/HOP;
  printPeakResults();
  Serial.printf("FILE_DONE,%s,%u,%llu,expected_frames=%llu,frontend_mean_us=%llu,frame_min_us=%llu,frame_max_us=%llu,min_heap=%u,min_psram=%u\n",
    fileId,(unsigned)frameCount,(unsigned long long)samplesSeen,(unsigned long long)expectedFrames,
    (unsigned long long)(frameCount?totalFrontendUs/frameCount:0),(unsigned long long)(frameCount?minFrameTotalUs:0),
    (unsigned long long)maxFrameTotalUs,(unsigned)minHeapAtStart,(unsigned)minPsramAtStart);
  printMemory();if(eventScores){free(eventScores);eventScores=nullptr;scoreCapacity=0;}
}

void handleCommand(const String &cmd){
  if(cmd=="PING"){Serial.println("PONG");return;}
  if(cmd=="ABORT"){streamActive=false;expectedSamples=samplesSeen=0;resetFile();Serial.println("ABORT_ACK");return;}
  if(cmd.startsWith("THRESH ")){char name[32]="";float p=0;unsigned distance=1;
    if(sscanf(cmd.c_str()+7,"%31s %f %u",name,&p,&distance)!=3){Serial.println("ERROR,THRESH_HEADER");return;}
    const int index=profileIndex(name);if(index<0){Serial.println("ERROR,THRESH_PROFILE");return;}
    peakProminence[index]=p;peakDistance[index]=(uint16_t)min(distance,65535u);peakConfigured[index]=true;
    Serial.printf("THRESH_ACK,%s,%.8g,%u\n",name,(double)p,(unsigned)peakDistance[index]);return;}
  if(cmd.startsWith("FILE ")){
    if(streamActive){Serial.println("ERROR,FILE_BUSY");return;}
    unsigned long long samples=0;float mean=0;int truth=-1;char id[64]="",split[24]="unknown";
    const int count=sscanf(cmd.c_str()+5,"%63s %llu %f %d %23s",id,&samples,&mean,&truth,split);
    if(count<5||samples<NFFT){Serial.println("ERROR,FILE_HEADER");return;}
    strlcpy(fileId,id,sizeof(fileId));strlcpy(dataSplit,split,sizeof(dataSplit));expectedSamples=samples;fileMean=mean;truthCount=truth;resetFile();
    scoreCapacity=expectedSamples<NFFT?0:1+(expectedSamples-NFFT)/HOP;
    eventScores=(float*)heap_caps_malloc(scoreCapacity*N_PROFILES*sizeof(float),MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
    if(!eventScores)eventScores=(float*)malloc(scoreCapacity*N_PROFILES*sizeof(float));
    if(!eventScores){Serial.println("ERROR,NO_SCORE_MEMORY");scoreCapacity=0;return;}
    streamActive=true;
    Serial.printf("FILE_ACK,%s,%llu,%d,%s\n",fileId,(unsigned long long)expectedSamples,truthCount,dataSplit);return;
  }
  if(cmd.startsWith("DATA ")){
    if(!streamActive){Serial.println("ERROR,NO_FILE");return;}
    const size_t count=strtoul(cmd.c_str()+5,nullptr,10);
    if(!count||count>BLOCK||samplesSeen+count>expectedSamples||!Serial.readBytes((char*)rx,count*sizeof(float))){Serial.println("ERROR,DATA_BLOCK");streamActive=false;return;}
    const float *values=(const float*)rx;for(size_t i=0;i<count;i++){addSample(values[i]);samplesSeen++;}
    Serial.printf("ACK,%llu\n",(unsigned long long)samplesSeen);return;
  }
  if(cmd=="END"){finishFile();return;}
  Serial.println("ERROR,UNKNOWN_COMMAND");
}
}

void setup(){
  Serial.begin(2000000);Serial.setTimeout(30000);
  for(size_t n=0;n<NFFT;n++)windowFn[n]=.5f-.5f*cosf(2*PI_F*(float)n/NFFT);
  buildGram(AVS_OLD_W,NBINS,gramOld);buildGram(AVS_MEL64_W,64,gram64);buildGram(AVS_MEL128_W,128,gram128);
  ready=true;Serial.println("AVS_DATASET_READY,profiles=14,format=float32le,streaming=1");printMemory();
}
void loop(){if(ready&&Serial.available()){String cmd=Serial.readStringUntil('\n');cmd.trim();handleCommand(cmd);}}
