// Command serve hosts the built demo on localhost. It is a convenience for
// development only: demo/dist is a plain static folder and any web server
// will do.
//
//	go run ./demo/serve            # http://localhost:8080
//	go run ./demo/serve -addr :9000 -dir demo/dist
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
)

func main() {
	addr := flag.String("addr", "localhost:8080", "address to listen on")
	dir := flag.String("dir", "demo/dist", "folder to serve")
	flag.Parse()

	if _, err := os.Stat(filepath.Join(*dir, "main.wasm")); err != nil {
		log.Fatalf("%s has no main.wasm: run ./demo/build.sh first", *dir)
	}

	fmt.Printf("serving %s at http://%s\n", *dir, *addr)
	log.Fatal(http.ListenAndServe(*addr, http.FileServer(http.Dir(*dir))))
}
