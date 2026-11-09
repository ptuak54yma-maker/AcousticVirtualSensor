set(AVS_DSP_SOURCES "avs_pipeline.cpp" "avs_fft.cpp")
set(AVS_DSP_REQUIRES)
if(CONFIG_AVS_USE_ESP_DSP)
    set(AVS_DSP_SOURCES "avs_pipeline_espdsp.cpp")
    list(APPEND AVS_DSP_REQUIRES esp-dsp)
endif()

idf_component_register(
    SRCS
        "app_main.cpp"
        ${AVS_DSP_SOURCES}
        "avs_peak.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_i2s esp_timer ${AVS_DSP_REQUIRES}
)

target_compile_features(${COMPONENT_LIB} PUBLIC cxx_std_17)
