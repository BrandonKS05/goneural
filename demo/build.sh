#!/bin/sh
# Builds the demo into demo/dist: a static folder holding the page, the
# library compiled to WebAssembly, and the wasm_exec.js loader that ships
# with the Go toolchain doing the compiling (the two must match).
set -eu

cd "$(dirname "$0")/.."
out=demo/dist

# The loader moved from misc/wasm to lib/wasm in Go 1.24.
goroot="$(go env GOROOT)"
loader=""
for candidate in "$goroot/lib/wasm/wasm_exec.js" "$goroot/misc/wasm/wasm_exec.js"; do
	if [ -f "$candidate" ]; then
		loader="$candidate"
		break
	fi
done
if [ -z "$loader" ]; then
	echo "build: no wasm_exec.js under $goroot" >&2
	exit 1
fi

rm -rf "$out"
mkdir -p "$out"

cp demo/web/index.html demo/web/style.css demo/web/app.js "$out/"
cp "$loader" "$out/wasm_exec.js"
GOOS=js GOARCH=wasm go build -trimpath -ldflags="-s -w" -o "$out/main.wasm" ./demo/wasm

echo "built $out with $(go env GOVERSION):"
ls -l "$out" | awk 'NR > 1 { printf "  %9d  %s\n", $5, $NF }'
