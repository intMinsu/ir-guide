# TensorRT wrapper for classification models
import numpy as np

class TRTClassifier:
    def __init__(self, engine_path: str):
        # compat shim for older TRT bindings that reference np.bool
        if not hasattr(np, "bool"):
            np.bool = np.bool_

        import tensorrt as trt
        import pycuda.driver as cuda
        try:
            import pycuda.autoprimaryctx  # attach primary CUDA context (shared with TRT/jetson_utils)
            print("[trt] using CUDA primary context for PyCUDA interop")
        except Exception:
            import pycuda.autoinit  # noqa: F401

        self.trt = trt
        self.cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create execution context")

        self._use_v3 = (
            hasattr(self.context, "set_tensor_address")
            and hasattr(self.context, "execute_async_v3")
            and hasattr(self.engine, "__iter__")
        )

        if self._use_v3:
            io_names = [n for n in self.engine]
            inputs  = [n for n in io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            outputs = [n for n in io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            assert len(inputs) == 1 and len(outputs) == 1, f"expect 1 input/1 output, got {inputs} -> {outputs}"
            self.input_name  = inputs[0]
            self.output_name = outputs[0]
            self.input_shape  = tuple(self.engine.get_tensor_shape(self.input_name))
            self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
            self.input_dtype  = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
            self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
        else:
            self.input_idx, self.output_idx = None, None
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i):
                    self.input_idx = i
                else:
                    self.output_idx = i
            assert self.input_idx is not None and self.output_idx is not None
            self.input_name   = self.engine.get_binding_name(self.input_idx)
            self.output_name  = self.engine.get_binding_name(self.output_idx)
            self.input_shape  = tuple(self.engine.get_binding_shape(self.input_idx))
            self.output_shape = tuple(self.engine.get_binding_shape(self.output_idx))
            self.input_dtype  = trt.nptype(self.engine.get_binding_dtype(self.input_idx))
            self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_idx))

        self._default_shape = (1, 3, 224, 224)
        if -1 in self.input_shape:
            self._set_shape(self._default_shape)
            self.input_shape = self._default_shape

        self._alloc_device_buffers(self.input_shape, self.output_shape)
        self.stream = self.cuda.Stream()

    def _set_shape(self, shape):
        try:
            if self._use_v3 and hasattr(self.context, "set_input_shape"):
                self.context.set_input_shape(self.input_name, list(shape))
            elif hasattr(self.context, "set_binding_shape"):
                self.context.set_binding_shape(self.input_idx, list(shape))
        except Exception:
            pass  # ok for static engines

    def _alloc_device_buffers(self, in_shape, out_shape):
        in_bytes  = int(np.prod(in_shape))  * np.dtype(self.input_dtype).itemsize
        out_bytes = int(np.prod(out_shape)) * np.dtype(self.output_dtype).itemsize
        self.d_input  = self.cuda.mem_alloc(in_bytes)
        self.d_output = self.cuda.mem_alloc(out_bytes)
        self.h_output = np.empty(out_shape, dtype=self.output_dtype)

    def infer(self, np_input: np.ndarray) -> np.ndarray:
        if np_input.dtype != self.input_dtype:
            np_input = np_input.astype(self.input_dtype, copy=False)
        if not np_input.flags.c_contiguous:
            np_input = np.ascontiguousarray(np_input)

        use_stream = True
        try:
            self.cuda.memcpy_htod_async(self.d_input, np_input, self.stream)
        except Exception:
            self.cuda.memcpy_htod(self.d_input, np_input)
            use_stream = False

        ok = False
        if self._use_v3:
            self.context.set_tensor_address(self.input_name,  int(self.d_input))
            self.context.set_tensor_address(self.output_name, int(self.d_output))
            try:
                ok = self.context.execute_async_v3(self.stream.handle if use_stream else 0)
            except Exception:
                try: self.stream.synchronize()
                except Exception: pass
                ok = self.context.execute_async_v3(0)
        else:
            bindings = [0] * self.engine.num_bindings
            bindings[self.input_idx]  = int(self.d_input)
            bindings[self.output_idx] = int(self.d_output)
            try:
                ok = self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
            except Exception:
                ok = self.context.execute_v2(bindings)

        if not ok:
            raise RuntimeError("TensorRT execute_async failed")

        try:
            self.cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
        except Exception:
            self.cuda.memcpy_dtoh(self.h_output, self.d_output)

        return self.h_output.astype(np.float32, copy=False)
