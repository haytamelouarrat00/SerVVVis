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
        self._cam_node = None
        self._cam_intrinsics_key = None
        self._last_camera = None

    def _get_renderer(self, W, H):
        if self._renderer is None or self._renderer_size != (W, H):
            if self._renderer is not None:
                self._renderer.delete()
            self._renderer = pyrender.OffscreenRenderer(W, H)
            self._renderer_size = (W, H)
        return self._renderer

    def _get_cam_node(self, camera, pose):
        key = (camera.fx, camera.fy, camera.cx, camera.cy)
        if self._cam_node is None or self._cam_intrinsics_key != key:
            if self._cam_node is not None:
                self.scene.remove_node(self._cam_node)
            intrinsics = pyrender.IntrinsicsCamera(
                fx=camera.fx, fy=camera.fy,
                cx=camera.cx, cy=camera.cy,
                znear=0.01, zfar=100.0,
            )
            self._cam_node = self.scene.add(intrinsics, pose=pose)
            self._cam_intrinsics_key = key
        else:
            self.scene.set_pose(self._cam_node, pose)
        return self._cam_node

    def render(self, camera):
        self._last_camera = camera
        T_world_cam_gl = camera.T_world_cam @ _CV_TO_GL
        self._get_cam_node(camera, T_world_cam_gl)
        renderer = self._get_renderer(camera.W, camera.H)
        color, _ = renderer.render(self.scene, flags=pyrender.RenderFlags.NONE)
        return (color / 255.0).astype(np.float32)

    def render_depth(self, camera=None):
        if camera is None:
            camera = self._last_camera
        if camera is None:
            raise NotImplementedError("MeshScene.render_depth requires a camera")

        T_world_cam_gl = camera.T_world_cam @ _CV_TO_GL
        self._get_cam_node(camera, T_world_cam_gl)
        renderer = self._get_renderer(camera.W, camera.H)
        _, depth = renderer.render(self.scene, flags=pyrender.RenderFlags.NONE)
        return depth.astype(np.float32)

    def __del__(self):
        try:
            if self._renderer is not None:
                self._renderer.delete()
        except Exception:
            pass
