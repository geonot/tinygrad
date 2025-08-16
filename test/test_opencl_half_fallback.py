import unittest, os, platform
import numpy as np
from tinygrad.device import Device, is_dtype_supported
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer import ProgramSpec
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.engine.realize import CompiledRunner
from tinygrad.codegen import full_rewrite

# Determine GPU availability without forcing Device.DEFAULT
try:
  HAS_GPU = any(dev.split(":")[0] == "GPU" for dev in Device.get_available_devices())
except Exception:
  HAS_GPU = False

@unittest.skipUnless(HAS_GPU, "tests target OpenCL GPU backend")
class TestOpenCLHalfFallback(unittest.TestCase):
  def _run_store_where(self, cond: bool, bv: float, cv: float, dtype):
    # Build a tiny WHERE pipeline in UOps to emit kernel code near renderer.
    a = UOp(Ops.DEFINE_GLOBAL, dtype.ptr(), (), 0)
    idx = UOp.const(dtypes.int32, 0)
    cond_u = UOp.const(dtypes.bool, cond)
    b = UOp.const(dtype, bv)
    c = UOp.const(dtype, cv)
    out = UOp.store(a.index(idx), UOp.where(cond_u, b, c))
    uops = full_rewrite(UOp.sink(out), Device[Device.DEFAULT].renderer)
    # use engine to compile and run program
    src = Device[Device.DEFAULT].renderer.render(uops)
    ei = CompiledRunner(ProgramSpec("test", src, Device.DEFAULT, uops[-1], uops=uops))
    from tinygrad.device import Buffer
    outb = Buffer(Device.DEFAULT, 1, dtype).allocate()
    ei.exec([outb])
    ret = np.empty(1, _to_np_dtype(dtype))
    outb.copyout(ret.data)
    return ret[0]

  @unittest.skipIf(os.environ.get('CI') or platform.system() == 'Darwin', "skip on CI/OSX where gating differs")
  def test_dtype_supported_reflects_extension(self):
    # For GPU (OpenCL), half support should reflect cl_khr_fp16 presence
    exts = getattr(Device[Device.DEFAULT], 'device_exts', '')
    supported = is_dtype_supported(dtypes.half, device=Device.DEFAULT)
    self.assertEqual('cl_khr_fp16' in exts, supported)

  @unittest.skipUnless(HAS_GPU and not is_dtype_supported(dtypes.half, device=Device.DEFAULT), "skip when fp16 supported natively or no GPU")
  def test_where_half_select_behaves(self):
    # Ensure WHERE on half yields correct selection even without fp16 extension
    for cond in [False, True]:
      res = self._run_store_where(cond, np.float16(-3.0).item(), np.float16(5.0).item(), dtypes.half)
      np.testing.assert_equal(res, np.float16(-3.0 if cond else 5.0))

if __name__ == '__main__':
  unittest.main()
