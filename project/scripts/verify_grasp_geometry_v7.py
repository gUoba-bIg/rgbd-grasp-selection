"""Deterministic synthetic CPU geometry checks; never a dataset AP result."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / 'runtime/envs/grasp_eval'
OUTPUT = ROOT / 'results/P2/grasp_geometry_v7.json'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
sys.dont_write_bytecode = True


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_prior_report():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        previous = json.loads(OUTPUT.read_text(encoding='utf-8'))
        archive = OUTPUT.parent / 'grasp_eval_v7_attempts'
        archive.mkdir(exist_ok=True)
        index = len(list(archive.glob('geometry_*.json'))) + 1
        prior_path = archive / ('geometry_{:03d}.json'.format(index))
        with prior_path.open('x', encoding='utf-8') as stream:
            json.dump(previous, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())


def write_report(report):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, OUTPUT)


def main():
    archive_prior_report()
    report = {'schema': 7, 'status': 'running', 'synthetic_engineering_checks_only': True,
              'dataset_ap_computed': False, 'dataset_frames_read': 0, 'gpu_used': False,
              'checks': {}, 'failures': [], 'python': sys.version,
              'executable': sys.executable, 'script_sha256': sha256(__file__)}
    try:
        if Path(sys.prefix).resolve() != ENV.resolve():
            raise RuntimeError('Self-check must run inside isolated grasp_eval')
        import numpy as np
        import open3d as o3d
        import graspnetAPI
        from graspnetAPI import GraspGroup
        from graspnetAPI.utils.eval_utils import collision_detection, get_grasp_score, transform_points
        from graspnetAPI.utils.config import get_config
        from graspnetAPI.utils.dexnet.grasping.quality import PointGraspMetrics3D
        from graspnetAPI.utils.dexnet.grasping.contacts import Contact3D
        from graspnetAPI.utils.dexnet.grasping.grasp import ParallelJawPtGrasp3D
        from graspnetAPI.utils.dexnet.grasping.graspable_object import GraspableObject3D
        from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
        from graspnetAPI.utils.dexnet.grasping.meshpy.mesh import Mesh3D
        from graspnetAPI.utils.dexnet.grasping.meshpy.sdf import Sdf3D
        if not Path(graspnetAPI.__file__).resolve().is_relative_to(ENV.resolve()):
            raise RuntimeError('Evaluation API escaped isolated environment')
        if 'torch' in sys.modules or hasattr(np, 'int'):
            raise RuntimeError('Unexpected tensor runtime or global NumPy alias monkey-patch')
        np.random.seed(20261022)

        grasps = np.zeros((3, 17), dtype=np.float64)
        grasps[:, 0] = [.9, .8, .7]
        grasps[:, 1:4] = [.06, .02, .02]
        grasps[:, 4:13] = np.eye(3).reshape(1, 9)
        grasps[:, 13] = [0, .001, .08]
        kept = GraspGroup(grasps).nms(.03, np.pi / 6).grasp_group_array
        assert len(kept) == 2
        np.testing.assert_allclose(np.sort(kept[:, 0]), [.7, .9], atol=1e-12, rtol=0)
        report['checks']['official_graspgroup_compiled_nms'] = {'passed': True, 'input': 3, 'retained': 2}

        inner = np.array([[x, y, z] for x in (-.005, 0, .005)
                          for y in (-.01, 0, .01) for z in (-.003, 0, .003)])
        candidate = grasps[:1].copy()
        collision, empty = collision_detection([candidate], [inner], [None], [np.eye(4)], inner)
        assert not bool(collision[0][0]) and not bool(empty[0][0])
        obstacle = np.vstack([inner, [0, .035, 0]])
        collision_obstacle, empty_obstacle = collision_detection([candidate], [inner], [None], [np.eye(4)], obstacle)
        assert bool(collision_obstacle[0][0]) and not bool(empty_obstacle[0][0])
        collision_empty, empty_empty = collision_detection([candidate], [inner], [None], [np.eye(4)], inner[:2])
        assert bool(collision_empty[0][0]) and bool(empty_empty[0][0])
        _, _, dex = collision_detection([candidate], [inner], [None], [np.eye(4)], inner, return_dexgrasps=True)
        assert isinstance(dex[0][0], ParallelJawPtGrasp3D)
        np.testing.assert_allclose(dex[0][0].center, [.02, 0, 0], atol=1e-12)
        report['checks']['official_collision_and_dex_conversion'] = {
            'passed': True, 'free': False, 'finger_obstacle': True, 'empty_rejected': True,
            'inner_points': len(inner), 'empty_threshold': 10}

        transform = np.eye(4)
        transform[:3, 3] = [.1, -.2, .3]
        np.testing.assert_allclose(transform_points(inner, transform), inner + [.1, -.2, .3], atol=1e-12)
        report['checks']['rigid_transform_meters'] = {'passed': True}

        radius, resolution = .02, .001
        coords = np.linspace(-.04, .04, 81)
        grid = np.stack(np.meshgrid(coords, coords, coords, indexing='ij'), axis=-1)
        signed_distance = np.linalg.norm(grid, axis=-1) - radius
        sdf = Sdf3D(signed_distance, np.full(3, -.04), resolution)
        center_index = np.array([40, 40, 40])
        assert np.isfinite(sdf[center_index])
        np.testing.assert_allclose(sdf[center_index], -.02, atol=1e-12)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=12)
        vertices = np.asarray(sphere.vertices)
        triangles = np.asarray(sphere.triangles)
        mesh = Mesh3D(np.vstack([vertices, [1, 1, 1]]), triangles)
        assert mesh.remove_unreferenced_vertices() and len(mesh.vertices) == len(vertices)
        obj = GraspableObject3D(sdf, mesh, key='synthetic_sphere_not_dataset')
        left = Contact3D(obj, np.array([-radius, 0, 0]), np.array([1., 0, 0]))
        right = Contact3D(obj, np.array([radius, 0, 0]), np.array([-1., 0, 0]))
        top = Contact3D(obj, np.array([0, radius, 0]), np.array([0, -1., 0]))
        for contact in (left, right, top):
            assert contact.normal is not None and np.isfinite(contact.normal).all()
        positive = PointGraspMetrics3D.force_closure(left, right, .4)
        negative = PointGraspMetrics3D.force_closure(left, top, .4)
        identical = PointGraspMetrics3D.force_closure(left, left, .4)
        assert positive == 1 and negative == 0 and identical == 0
        report['checks']['sdf_mesh_alias_compatibility'] = {'passed': True, 'resolution_m': resolution, 'radius_m': radius}
        report['checks']['official_force_closure_contacts'] = {
            'passed': True, 'friction': .4, 'antipodal': int(positive), 'orthogonal': int(negative), 'identical_contact': int(identical)}

        params = get_config()['metrics']['force_closure']
        params['friction_coef'] = .4
        quality_config = GraspQualityConfigFactory.create_config(params)
        grasp = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(np.zeros(3), np.array([1., 0, 0]), .06))
        with_contacts = PointGraspMetrics3D.grasp_quality(grasp, obj, quality_config, contacts=[left, right])
        auto_contacts = PointGraspMetrics3D.grasp_quality(grasp, obj, quality_config)
        found, searched = grasp.close_fingers(obj, check_approach=False)
        report['checks']['automatic_contact_search_diagnostic'] = {
            'explicit_contact_quality': int(with_contacts), 'searched_contact_quality': int(auto_contacts),
            'contacts_found': bool(found), 'contacts': [None if contact is None else {
                'point': contact.point.tolist(), 'normal': None if contact.normal is None else contact.normal.tolist()}
                for contact in (searched or [])]}
        # Keep the observed axial failure: the upstream quadratic zero-crossing
        # helper explicitly rejects locally linear signed-distance samples.
        linear_crossing = Sdf3D.find_zero_crossing_quadratic(
            np.array([0., 0, 0]), .001, np.array([1., 0, 0]), 0.,
            np.array([2., 0, 0]), -.001)
        assert with_contacts == 1 and auto_contacts == 0 and not found and linear_crossing is None
        report['checks']['automatic_contact_search_diagnostic'].update({
            'known_upstream_limitation': True,
            'axial_sphere_expected_geometric_quality': 1,
            'linear_zero_crossing_returned_none': True,
            'interpretation': 'Upstream rejects a vanishing quadratic coefficient even when a linear zero exists; no scoring change applied'})
        # This explicitly revised, nondegenerate fixture covers a successful
        # automatic search without concealing the axial counterexample above.
        off_axis_grasp = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
            np.array([0., .003, 0]), np.array([1., 0, 0]), .06))
        nondegenerate_quality = PointGraspMetrics3D.grasp_quality(off_axis_grasp, obj, quality_config)
        found_offset, contacts_offset = off_axis_grasp.close_fingers(obj, check_approach=False)
        assert nondegenerate_quality == 1 and found_offset
        expected_contact_x = np.sqrt(radius ** 2 - .003 ** 2)
        np.testing.assert_allclose(
            np.sort([contact.point[0] for contact in contacts_offset]),
            [-expected_contact_x, expected_contact_x], atol=resolution, rtol=0)
        configs = {}
        for friction in (1.2, .8, .4, .2):
            configs[friction] = GraspQualityConfigFactory.create_config(dict(params, friction_coef=friction))
        score = get_grasp_score(off_axis_grasp, obj, [1.2, .8, .4, .2], configs)
        assert score == .2
        report['checks']['official_quality_contact_search_and_score'] = {
            'passed': True, 'fixture': 'sphere, grasp center shifted +0.003 m in y',
            'explicit_axial_contact_quality': int(with_contacts), 'nondegenerate_searched_contact_quality': int(nondegenerate_quality),
            'searched_contact_points': [contact.point.tolist() for contact in contacts_offset],
            'analytic_required_friction': .003 / expected_contact_x,
            'synthetic_required_friction_on_tested_grid': score,
            'note': 'Revised nondegenerate fixture after axial failure; not AP or real grasp success probability'}

        # Exercise the real cvxopt QP branch with known convex-hull cases.
        wrenches = np.concatenate([np.eye(6), -np.eye(6)], axis=1)
        qp_config = GraspQualityConfigFactory.create_config(dict(params, torque_scaling=1.))
        qp_positive = PointGraspMetrics3D.force_closure_qp(wrenches[:3], wrenches[3:], np.zeros((3, 12)), params=qp_config)
        one_sided = np.eye(6)
        qp_negative = PointGraspMetrics3D.force_closure_qp(one_sided[:3], one_sided[3:], np.zeros((3, 6)), params=qp_config)
        assert qp_positive == 1 and qp_negative == 0
        report['checks']['official_cvxopt_force_closure_qp'] = {'passed': True, 'symmetric_wrenches': int(qp_positive), 'one_sided': int(qp_negative)}
        report['packages'] = {item.metadata['Name']: item.version for item in importlib.metadata.distributions()}
        report['status'] = 'limited_engineering_ready_nondegenerate_paths_passed'
        report['all_geometry_reliable'] = False
        report['fixture_revision'] = {
            'reason': 'Initial axial sphere expected positive but upstream automatic contact search returned no contacts',
            'retained_counterexample': 'Axial sphere remains in this test and report; no evaluation algorithm was changed',
            'new_positive_fixture': 'Same sphere, grasp center shifted +0.003 m in y',
            'test_interface_corrections': ['Sdf3D indexing requires numpy array, not a Python tuple',
                                           'Direct force_closure_qp requires a config containing torque_scaling'],
            'prior_attempts': [{'file': str(path.relative_to(ROOT)), 'sha256': sha256(path),
                               'script_sha256': json.loads(path.read_text(encoding='utf-8')).get('script_sha256')}
                              for path in sorted((OUTPUT.parent / 'grasp_eval_v7_attempts').glob('geometry_*.json'))]}
        report['limitations'] = ['Known axial-sphere contact-search false negative from upstream quadratic interpolation; not fixed',
                                 'Synthetic tests only; no real object XML/PLY/OBJ/SDF dataset read',
                                 'No AP, calibration, official test or pretrained model loaded',
                                 'Not all upstream optional metrics or real-object degenerate cases covered',
                                 'Numerical compatibility does not yet establish benchmark reproduction']
    except Exception as error:
        report['status'] = 'failed'
        report['failures'].append({'error': type(error).__name__ + ': ' + str(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        write_report(report)


if __name__ == '__main__':
    main()
