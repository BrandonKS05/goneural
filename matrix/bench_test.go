package matrix

import "testing"

func randomMatrix(r, c int) Matrix {
	m := Zeros(r, c)
	m.Randomize(-1, 1)
	return m
}

// The 784-input shapes mirror the MNIST example, the library's heaviest
// real workload.

func BenchmarkMultiplySquare128(b *testing.B) {
	x := randomMatrix(128, 128)
	y := randomMatrix(128, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.Multiply(y)
	}
}

func BenchmarkMultiplyVec(b *testing.B) {
	w := randomMatrix(128, 784)
	a := randomMatrix(784, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.Multiply(a)
	}
}

func BenchmarkAddMatrix(b *testing.B) {
	x := randomMatrix(128, 784)
	y := randomMatrix(128, 784)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.AddMatrix(y)
	}
}

func BenchmarkTranspose(b *testing.B) {
	x := randomMatrix(128, 784)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.Transpose()
	}
}

func BenchmarkMap(b *testing.B) {
	x := randomMatrix(128, 784)
	f := func(val float64, _, _ int) float64 { return val * 2 }
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.Map(f)
	}
}
