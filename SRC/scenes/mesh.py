import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import trimesh
import pyrender

# OpenCV (x-right, y-down, z-forward) -> OpenGL (x-right, y-up, z-back)
_CV_TO_GL = np.diag([1, -1, -1, 1]).astype(np.float32)


class MeshScene:
    def __init__(self, ply_path):
        mesh = trimesh.load(ply_path, force='mesh')
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
        self.scene.add(pyrender_mesh)
        self._renderer = None
        self._renderer_size = None

    def _get_renderer(self, W, H):
        if self._renderer is None or self._renderer_size != (W, H):
            if self._renderer is not None:
                self._renderer.delete()
            self._renderer = pyrender.OffscreenRenderer(W, H)
            self._renderer_size = (W, H)
        return self._renderer

    def render(self, camera):
        intrinsics = pyrender.IntrinsicsCamera(
            fx=camera.fx, fy=camera.fy,
            cx=camera.cx, cy=camera.cy,
            znear=0.01, zfar=100.0
        )

        T_world_cam_gl = camera.T_world_cam @ _CV_TO_GL

        cam_node = self.scene.add(intrinsics, pose=T_world_cam_gl)
        renderer = self._get_renderer(camera.W, camera.H)
        color, depth = renderer.render(self.scene, flags=pyrender.RenderFlags.NONE)
        self.scene.remove_node(cam_node)

        return (color / 255.0).astype(np.float32)

    def __del__(self):
        try:
            if self._renderer is not None:
                self._renderer.delete()
        except Exception:
            pass
