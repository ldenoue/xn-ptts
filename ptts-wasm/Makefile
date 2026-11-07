.PHONY: build profiling clean

build:
	wasm-pack build --target web --release
	cp pkg/ptts_wasm.js pkg/ptts_wasm_bg.wasm www/
	cp www/index.html www/*.js www/*.wasm pkg/
	cp -R www/woof-runtime pkg/

profiling:
	wasm-pack build --target web --profiling
	cp pkg/ptts_wasm.js pkg/ptts_wasm_bg.wasm www/
	cp www/index.html www/*.js www/*.wasm pkg/
	cp -R www/woof-runtime pkg/

clean:
	rm -rf pkg
