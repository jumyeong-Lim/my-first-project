"""강체운동 제거와 기하학 회전 지표의 독립 검증."""
import unittest
import numpy as np
from verification.step16_integrated_elastic_conclusion import (
    XY,rigid_geometry,joint_distortion,geometry_rotation_fraction,physical_energy,
)


class IntegrationMetricTests(unittest.TestCase):
    def test_rigid_motion_removed(self):
        u=(rigid_geometry()@np.array([2.,-3.,.04])).reshape(12,2)
        self.assertLess(joint_distortion(u),1e-12)
        axial=np.column_stack((.1*XY[:,0],np.zeros(12)))
        self.assertAlmostEqual(joint_distortion(axial),.1*np.sqrt(45.),places=12)

    def test_geometric_translation_and_rotation(self):
        trans=np.tile((rigid_geometry()@[1.,2.,0.]),3)
        rot=np.tile((rigid_geometry()@[0.,0.,.1]),3)
        self.assertAlmostEqual(geometry_rotation_fraction(trans),0.,places=12)
        self.assertAlmostEqual(geometry_rotation_fraction(rot),1.,places=12)
        self.assertAlmostEqual(geometry_rotation_fraction(-7*rot),1.,places=12)

    def test_physical_energy_changes_with_reference_and_mass(self):
        weights=np.full(12,.01/12);weights[-1]+=.99
        u=np.tile((rigid_geometry()@[0.,0.,.1]),3)
        e=physical_energy(u,weights,1.)
        total=np.sum(np.tile(np.repeat(weights,2),3)*u**2)
        self.assertAlmostEqual(sum(e),total,places=12)
        self.assertLess(e[2]/total,.1)
        self.assertGreater(geometry_rotation_fraction(u),.99)


if __name__=="__main__":unittest.main()
