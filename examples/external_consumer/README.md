# External consumer example

这是一个演示如何把 pc-nsf-hifigan 作为第三方库集成的极简工程（"合入式"
add_subdirectory 模式）。

```bash
cmake -S examples/external_consumer -B /tmp/consumer-build -D PCNSF_BUILD_TESTS=OFF -D PCNSF_BUILD_CLI=OFF
cmake --build /tmp/consumer-build -j
/tmp/consumer-build/consumer hifigan.gguf mel.bin f0.bin out.wav
```

用真实 GGUF 执行即可。consumer 只依赖 `pc_nsf_hifigan::pc_nsf_hifigan`
target，源码仅调 `HifiganModel` + `hifigan_run`，是下游 vocoder bridge
（如 DiffSinger 风格 C++ 管线）的 API 模板。

> install/export 模式与 add_subdirectory 等价。OpenUtau 插件的做法是把本仓库
> 当作 opudep 拉入，再用两个 CMake target 之一链接。
