idf_component_register(
    SRCS
        "app_main.cpp"
        "avs_pipeline.cpp"
        "avs_fft.cpp"
        "avs_peak.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_i2s esp_timer
)

target_compile_features(${COMPONENT_LIB} PUBLIC cxx_std_17)
