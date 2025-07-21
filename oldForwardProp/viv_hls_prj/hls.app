<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" name="viv_hls_prj" top="top">
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="false" ldflags="" mflags=""/>
    </Simulation>
    <files>
        <file name="input_bpsk.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="../../TestBench.cpp" sc="0" tb="1" cflags="-Wno-unknown-pragmas" csimflags="" blackbox="false"/>
        <file name="TestBench.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="model_consts.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="model_impl.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="params.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="input.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="utils.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="FullyConnected.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="pooling.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="conv.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="core.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="cnn.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
    </files>
    <solutions>
        <solution name="cnn_synth" status=""/>
    </solutions>
</AutoPilot:project>

