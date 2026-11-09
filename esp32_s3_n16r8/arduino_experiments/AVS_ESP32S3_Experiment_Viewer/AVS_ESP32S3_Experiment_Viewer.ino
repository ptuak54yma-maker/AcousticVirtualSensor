#include <Arduino.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "esp_timer.h"
#include "experiment_models.h"  // Generate with ../../tools/export_arduino_experiment_models.py

namespace {
constexpr size_t NFFT = 2048, HOP = 1024, NBINS = 1025, K = 48, BLOCK = 512;
constexpr size_t MAX_FRAMES = 4096;
constexpr float PI_F = 3.14159265358979323846f;
struct Profile { const char *name; uint8_t kind; uint16_t mel; uint16_t iterations; };
// kind: 0=legacy log-magnitude + coordinate NNLS, 1=FISTA, 2=coordinate NNLS.
Profile profile{"m64_fista512", 1, 64, 512};
float windowFn[NFFT], realPart[NFFT], imagPart[NFFT], frame[NFFT];
float magnitude[NBINS], features[NBINS];
float gram[K*K], rhs[K], h[K], y[K], nextH[K];
float envelope[MAX_FRAMES]; uint32_t frameUs[MAX_FRAMES];
uint8_t rx[BLOCK*sizeof(float)];
size_t frameFill=0, hopFill=0, frameCount=0; uint64_t samplesExpected=0, samplesSeen=0;
float fileMean=0; bool stream=false, ready=false;

void fft() {
  for (size_t i=1,j=0;i<NFFT;i++) { size_t bit=NFFT>>1; for(;j&bit;bit>>=1) j^=bit; j^=bit;
    if(i<j){std::swap(realPart[i],realPart[j]);std::swap(imagPart[i],imagPart[j]);}}
  for(size_t len=2;len<=NFFT;len<<=1){float a=-2*PI_F/len, wrs=cosf(a), wis=sinf(a);
    for(size_t base=0;base<NFFT;base+=len){float wr=1,wi=0;for(size_t j=0;j<len/2;j++){
      size_t u=base+j,v=u+len/2;float vr=realPart[v]*wr-imagPart[v]*wi,vi=realPart[v]*wi+imagPart[v]*wr;
      float ur=realPart[u],ui=imagPart[u];realPart[u]=ur+vr;imagPart[u]=ui+vi;realPart[v]=ur-vr;imagPart[v]=ui-vi;
      float t=wr* wrs-wi*wis;wi=wr*wis+wi*wrs;wr=t;}}}
}
const float *dictW() {
  if(profile.kind==0) return AVS_OLD_W;
  return profile.mel==64 ? AVS_MEL64_W : AVS_MEL128_W;
}
size_t featureRows(){return profile.kind==0?NBINS:profile.mel;}
void project(const float *mag,float *v) {
  if(profile.kind==0){for(size_t f=0;f<NBINS;f++)v[f]=log1pf(mag[f]);return;}
  const uint16_t *offs=profile.mel==64?AVS_MEL64_OFFSETS:AVS_MEL128_OFFSETS;
  const uint16_t *idx=profile.mel==64?AVS_MEL64_BIN_INDEX:AVS_MEL128_BIN_INDEX;
  const float *weights=profile.mel==64?AVS_MEL64_WEIGHTS:AVS_MEL128_WEIGHTS;
  const float ref=profile.mel==64?AVS_MEL64_REFERENCE:AVS_MEL128_REFERENCE;
  for(size_t m=0;m<profile.mel;m++){float sum=0;for(uint16_t q=offs[m];q<offs[m+1];q++)sum+=weights[q]*mag[idx[q]];v[m]=log1pf(sum/ref);}
}
void initGram(){const float *w=dictW();size_t F=featureRows();memset(gram,0,sizeof(gram));
  for(size_t i=0;i<K;i++)for(size_t j=0;j<K;j++){float s=0;for(size_t f=0;f<F;f++)s+=w[f*K+i]*w[f*K+j];gram[i*K+j]=s;}}
void solve(const float *v){const float *w=dictW();size_t F=featureRows();
  for(size_t j=0;j<K;j++){float s=0;for(size_t f=0;f<F;f++)s+=w[f*K+j]*v[f];rhs[j]=s;}
  memset(h,0,sizeof(h));
  if(profile.kind==0 || profile.kind==2){ // Legacy-compatible nonnegative coordinate descent.
    for(unsigned it=0;it<80;it++)for(size_t j=0;j<K;j++){float residual=rhs[j];for(size_t k=0;k<K;k++)if(k!=j)residual-=gram[j*K+k]*h[k];
      float d=gram[j*K+j];h[j]=d>1e-20f?fmaxf(0,residual/d):0;} return;}
  memcpy(y,h,sizeof(h)); float t=1, L=0;
  for(size_t i=0;i<K;i++){float rs=0;for(size_t j=0;j<K;j++)rs+=gram[i*K+j];L=fmaxf(L,rs);} if(L<=0)return;
  for(unsigned it=0;it<profile.iterations;it++){float tn=0.5f*(1+sqrtf(1+4*t*t));
    for(size_t j=0;j<K;j++){float gy=0;for(size_t k=0;k<K;k++)gy+=gram[j*K+k]*y[k];nextH[j]=fmaxf(0,y[j]-(gy-rhs[j])/L);}
    for(size_t j=0;j<K;j++)y[j]=nextH[j]+((t-1)/tn)*(nextH[j]-h[j]);memcpy(h,nextH,sizeof(h));t=tn;}
}
const char *profileId(){static char id[32];
  if(profile.kind==0)snprintf(id,sizeof(id),"legacy_nnls");
  else if(profile.kind==1)snprintf(id,sizeof(id),"m%u_fista%u",(unsigned)profile.mel,(unsigned)profile.iterations);
  else snprintf(id,sizeof(id),"m%u_nnls",(unsigned)profile.mel);
  return id;
}
void processFrame(){if(frameCount>=MAX_FRAMES)return;uint32_t start=(uint32_t)esp_timer_get_time();
  for(size_t n=0;n<NFFT;n++){realPart[n]=frame[n]*windowFn[n];imagPart[n]=0;}fft();
  for(size_t f=0;f<NBINS;f++)magnitude[f]=sqrtf(realPart[f]*realPart[f]+imagPart[f]*imagPart[f]);
  project(magnitude,features);solve(features);float score=0;for(size_t j=0;j<24;j++)score+=h[j];
  envelope[frameCount]=score;frameUs[frameCount]=(uint32_t)esp_timer_get_time()-start;frameCount++;
}
void addSample(float x){x-=fileMean;if(frameFill<NFFT){frame[frameFill++]=x;if(frameFill==NFFT)processFrame();return;}
  frame[HOP+hopFill++]=x;if(hopFill==HOP){memmove(frame,frame+HOP,(NFFT-HOP)*sizeof(float));hopFill=0;processFrame();}}
void printResults(){Serial.println("RESULTS_BEGIN");uint64_t totalUs=0;uint32_t maxUs=0;float scoreSum=0,scoreMin=INFINITY,scoreMax=0;
  for(size_t i=0;i<frameCount;i++){Serial.printf("ROW,%s,%u,%.6f,%.9g,%lu\n",profileId(),(unsigned)i,(double)(i*HOP)/(double)44100,(double)envelope[i],(unsigned long)frameUs[i]);
    totalUs+=frameUs[i];maxUs=max(maxUs,frameUs[i]);scoreSum+=envelope[i];scoreMin=fminf(scoreMin,envelope[i]);scoreMax=fmaxf(scoreMax,envelope[i]);}
  Serial.printf("SUMMARY,%s,frames=%u,score_mean=%.9g,score_min=%.9g,score_max=%.9g,mean_us=%llu,max_us=%lu\n",profileId(),(unsigned)frameCount,
    frameCount?(double)(scoreSum/frameCount):0.0,(double)(frameCount?scoreMin:0.0f),(double)scoreMax,
    (unsigned long long)(frameCount?totalUs/frameCount:0),(unsigned long)maxUs);
  Serial.printf("DONE,%s,%u,%u,%llu\n",profileId(),(unsigned)profile.iterations,(unsigned)frameCount,(unsigned long long)samplesSeen);}
void handle(const String &cmd){if(cmd.startsWith("PROFILE ")){String p=cmd.substring(8);if(p=="legacy_nnls"){profile={"legacy_nnls",0,0,0};}
  else if(p=="m64_nnls"){profile={"m64_nnls",2,64,0};} else if(p=="m128_nnls"){profile={"m128_nnls",2,128,0};}
  else if(p.startsWith("m64_fista")){unsigned n=p.substring(9).toInt();if(n==0)n=512;profile={"m64_fista",1,64,(uint16_t)min(n,65535u)};}
  else if(p=="m128_fista512"){profile={"m128_fista",1,128,512};} else {Serial.println("ERROR PROFILE");return;}
  initGram();Serial.printf("PROFILE_OK,%s,features=%u,iterations=%u,solver=%s\n",profileId(),(unsigned)featureRows(),(unsigned)profile.iterations,profile.kind==1?"FISTA":"NNLS_CD");return;}
  if(cmd.startsWith("START ")){char *end=nullptr;samplesExpected=strtoull(cmd.c_str()+6,&end,10);fileMean=end?strtof(end,nullptr):0;samplesSeen=0;frameCount=0;frameFill=0;hopFill=0;stream=true;Serial.println("ACK_START");return;}
  if(cmd.startsWith("DATA ")){if(!stream){Serial.println("ERROR NO_STREAM");return;}size_t count=strtoul(cmd.c_str()+5,nullptr,10);if(!count||count>BLOCK||!Serial.readBytes((char*)rx,count*4)){Serial.println("ERROR DATA");stream=false;return;}
    float *vals=(float*)rx;for(size_t i=0;i<count;i++){addSample(vals[i]);samplesSeen++;}Serial.printf("ACK %llu\n",(unsigned long long)samplesSeen);return;}
  if(cmd=="END"){if(!stream||samplesSeen!=samplesExpected){Serial.println("ERROR END_COUNT");stream=false;return;}stream=false;printResults();return;}
  if(cmd=="PING")Serial.println("PONG");else Serial.println("ERROR COMMAND");}
}
void setup(){Serial.begin(2000000);Serial.setTimeout(30000);for(size_t n=0;n<NFFT;n++)windowFn[n]=0.5f-0.5f*cosf(2*PI_F*n/NFFT);initGram();ready=true;Serial.println("AVS_EXPERIMENT_READY format=float32le max_block=512");}
void loop(){if(ready&&Serial.available()){String cmd=Serial.readStringUntil('\n');cmd.trim();handle(cmd);}}
