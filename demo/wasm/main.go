//go:build js && wasm

// Command wasm exposes the stepper to a web page. It installs one global
// object, goneuralDemo, whose methods the page calls; every number the
// page shows is read from a snapshot returned here.
//
//	goneuralDemo.reset(word, seed) -> snapshot | {error}
//	goneuralDemo.step()            -> snapshot   one stage
//	goneuralDemo.stepPair(n = 1)   -> snapshot   through the next n updates
//	goneuralDemo.snapshot()        -> snapshot
//	goneuralDemo.sample()          -> string
//
// Build with GOOS=js GOARCH=wasm; see demo/build.sh.
package main

import (
	"fmt"
	"syscall/js"

	"github.com/BrandonKS05/goneural/demo/stepper"
)

// maxStepsPerCall bounds how long one stepPair call can hold the page's
// only thread.
const maxStepsPerCall = 1000

var current *stepper.Stepper

func main() {
	js.Global().Set("goneuralDemo", js.ValueOf(map[string]any{
		"reset": guarded(func(args []js.Value) any {
			if len(args) < 2 || args[0].Type() != js.TypeString || args[1].Type() != js.TypeNumber {
				return failure("reset needs a word and a numeric seed")
			}
			s, err := stepper.New(args[0].String(), int64(args[1].Float()))
			if err != nil {
				return failure(err.Error())
			}
			current = s
			return snapshot()
		}),
		"step": guarded(func([]js.Value) any {
			if current == nil {
				return failure("no model yet: call reset first")
			}
			current.Step()
			return snapshot()
		}),
		"stepPair": guarded(func(args []js.Value) any {
			if current == nil {
				return failure("no model yet: call reset first")
			}
			// An optional count runs several whole steps in one call, so
			// a page whose timer fires late can catch up without paying
			// for a snapshot per step.
			count := 1
			if len(args) > 0 && args[0].Type() == js.TypeNumber {
				count = min(max(args[0].Int(), 1), maxStepsPerCall)
			}
			for i := 0; i < count; i++ {
				current.StepPair()
			}
			return snapshot()
		}),
		"snapshot": guarded(func([]js.Value) any {
			if current == nil {
				return failure("no model yet: call reset first")
			}
			return snapshot()
		}),
		"sample": guarded(func([]js.Value) any {
			if current == nil {
				return ""
			}
			return current.Sample()
		}),
		"config": map[string]any{
			"hiddenUnits":     stepper.HiddenUnits,
			"learningRate":    stepper.LearningRate,
			"maxWordLength":   stepper.MaxWordLength,
			"maxSampleLength": stepper.MaxSampleLength,
		},
	}))

	// Tell the page the functions exist, then stay alive to serve them: a
	// Go program that returns from main tears its exports down with it.
	if ready := js.Global().Get("onGoneuralReady"); ready.Type() == js.TypeFunction {
		ready.Invoke()
	}
	select {}
}

// guarded wraps a handler so a panic inside the library reaches the page
// as an {error} value instead of killing the program.
func guarded(handler func(args []js.Value) any) js.Func {
	return js.FuncOf(func(_ js.Value, args []js.Value) (result any) {
		defer func() {
			if r := recover(); r != nil {
				result = failure(fmt.Sprint(r))
			}
		}()
		return handler(args)
	})
}

func failure(message string) any {
	return map[string]any{"error": message}
}

// snapshot converts the stepper's state into plain JavaScript objects and
// arrays.
func snapshot() any {
	s := current.Snapshot()

	pairs := make([]any, len(s.Pairs))
	for i, p := range s.Pairs {
		pairs[i] = map[string]any{"input": p.Input, "target": p.Target}
	}

	out := map[string]any{
		"word":          s.Word,
		"vocab":         stringsOf(s.Vocab),
		"pairs":         pairs,
		"stage":         s.Stage.String(),
		"updates":       s.Updates,
		"pair":          s.Pair,
		"input":         s.Input,
		"target":        s.Target,
		"hidden":        vector(s.Hidden),
		"probabilities": vector(s.Probabilities),
		"loss":          nil,
		"weights":       parameters(s.Weights),
		"gradients":     parameters(s.Gradients),
		"lastChange":    nil,
		"averageLoss":   s.AverageLoss,
		"bestLoss":      s.BestLoss,
	}

	if s.HasLoss {
		out["loss"] = s.Loss
	}
	if c := s.LastChange; c != nil {
		out["lastChange"] = map[string]any{
			"layer": c.Layer,
			"from":  c.From,
			"to":    c.To,
			"old":   c.Old,
			"new":   c.New,
		}
	}

	return out
}

func parameters(p stepper.Parameters) any {
	return map[string]any{
		"inputHidden":  grid(p.InputHidden),
		"hiddenBias":   vector(p.HiddenBias),
		"hiddenOutput": grid(p.HiddenOutput),
		"outputBias":   vector(p.OutputBias),
	}
}

func vector(values []float64) any {
	if values == nil {
		return nil
	}
	out := make([]any, len(values))
	for i, v := range values {
		out[i] = v
	}
	return out
}

func grid(values [][]float64) any {
	out := make([]any, len(values))
	for i, row := range values {
		out[i] = vector(row)
	}
	return out
}

func stringsOf(values []string) any {
	out := make([]any, len(values))
	for i, v := range values {
		out[i] = v
	}
	return out
}
