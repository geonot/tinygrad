import unittest
import numpy as np
from tinygrad.device import Device, is_dtype_supported
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer import ProgramSpec
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.engine.realize import CompiledRunner
from tinygrad.codegen import full_rewrite

try:
  HAS_GPU = any(dev.split(":")[0] == "GPU" for dev in Device.get_available_devices())
except Exception:
  HAS_GPU = False

@unittest.skipUnless(HAS_GPU, "tests target OpenCL GPU backend")
class TestOpenCLHalfOps(unittest.TestCase):
  def _run_unary(self, op: Ops, v: float):
    dtype = dtypes.half
    a = UOp(Ops.DEFINE_GLOBAL, dtype.ptr(), (), 0)
    idx = UOp.const(dtypes.int32, 0)
    x = UOp.const(dtype, v)
    out = UOp.store(a.index(idx), UOp(op, dtype, (x,), None))
    uops = full_rewrite(UOp.sink(out), Device[Device.DEFAULT].renderer)
    src = Device[Device.DEFAULT].renderer.render(uops)
    ei = CompiledRunner(ProgramSpec("test", src, Device.DEFAULT, uops[-1], uops=uops))
    from tinygrad.device import Buffer
    outb = Buffer(Device.DEFAULT, 1, dtype).allocate()
    ei.exec([outb])
    ret = np.empty(1, _to_np_dtype(dtype))
    outb.copyout(ret.data)
    return ret[0]

  def _run_binary(self, op: Ops, a: float, b: float):
    dtype = dtypes.half
    o = UOp(Ops.DEFINE_GLOBAL, dtype.ptr(), (), 0)
    idx = UOp.const(dtypes.int32, 0)
    x = UOp.const(dtype, a)
    y = UOp.const(dtype, b)
    out = UOp.store(o.index(idx), UOp(op, dtype, (x, y), None))
    uops = full_rewrite(UOp.sink(out), Device[Device.DEFAULT].renderer)
    src = Device[Device.DEFAULT].renderer.render(uops)
    ei = CompiledRunner(ProgramSpec("test", src, Device.DEFAULT, uops[-1], uops=uops))
    from tinygrad.device import Buffer
    outb = Buffer(Device.DEFAULT, 1, dtype).allocate()
    ei.exec([outb])
    ret = np.empty(1, _to_np_dtype(dtype))
    outb.copyout(ret.data)
    return ret[0]

  @unittest.skipUnless(HAS_GPU and not is_dtype_supported(dtypes.half, device=Device.DEFAULT), "skip when native fp16 is available or no GPU")
  def test_add_half_upcast_path(self):
    # Validate that half add equals numpy half add
    for a, b in [(-3.0, 2.0), (1.5, 0.25), (65504.0, 0.0)]:
      expected = np.float16(a) + np.float16(b)
      got = self._run_binary(Ops.ADD, np.float16(a).item(), np.float16(b).item())
      np.testing.assert_equal(got, expected)

  @unittest.skipUnless(HAS_GPU and not is_dtype_supported(dtypes.half, device=Device.DEFAULT), "skip when native fp16 is available or no GPU")
  def test_cast_roundtrip(self):
    # float -> half -> float roundtrip (approx equality within half precision)
    for v in [-3.125, -0.0, 0.0, 1.0, 1.333, 65504.0]:
      # Build roundtrip: CONST(float) -> CAST(half) -> CAST(float)
      a = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
      idx = UOp.const(dtypes.int32, 0)
      f = UOp.const(dtypes.float, float(v))
      h = UOp.cast(dtypes.half, f)
      back = UOp.cast(dtypes.float, h)
      out = UOp.store(a.index(idx), back)
      uops = full_rewrite(UOp.sink(out), Device[Device.DEFAULT].renderer)
      src = Device[Device.DEFAULT].renderer.render(uops)
      ei = CompiledRunner(ProgramSpec("test", src, Device.DEFAULT, uops[-1], uops=uops))
      from tinygrad.device import Buffer
      outb = Buffer(Device.DEFAULT, 1, dtypes.float).allocate()
      ei.exec([outb])
      ret = np.empty(1, _to_np_dtype(dtypes.float))
      outb.copyout(ret.data)
      # Compare to numpy fp16 roundtrip reference
      np.testing.assert_allclose(ret[0], np.float16(v).astype(np.float32), rtol=0, atol=0)

if __name__ == '__main__':
  unittest.main()
