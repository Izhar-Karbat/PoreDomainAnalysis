import numpy as np
import os
import pytest

import MDAnalysis as mda
from pore_analysis.modules.core_analysis.core import analyze_trajectory

class FakeResidue:
    def __init__(self, resname, resid):
        self.resname = resname
        self.resid = resid

class FakeAtomGroup:
    def __init__(self, residues=None, positions=None):
        self.residues = residues or []
        # positions: for per-frame, positions[frame][key]
        self._positions = positions or {}
    def __len__(self):
        # length for initial chain selection
        return len(self.residues)
    @property
    def residues(self):
        return self._residues
    @residues.setter
    def residues(self, val):
        self._residues = val
    def positions(self):
        # not used
        return None
    @property
    def positions(self):
        # return last frame positions if called outside trajectory loop
        return self._positions.get('static', np.array([[0,0,0]]))
    def center_of_mass(self):
        # not used in control branch
        return np.array([0,0,0])

class FakeTS:
    def __init__(self, frame):
        self.frame = frame

class FakeUniverse:
    def __init__(self, psf, dcd):
        # 3 frames
        self.trajectory = [FakeTS(i) for i in range(3)]
        # create chain_A_atoms residues GYG
        self._chainA = FakeAtomGroup(
            residues=[
                FakeResidue('GLY', 10),
                FakeResidue('TYR', 11),
                FakeResidue('GLY', 12),
            ]
        )
        # positions per frame for PROA and PROC
        # frame i: PROA at (0,0,0), PROC at (i+1,0,0)
        self._positions = {
            ('PROA', 12): [np.array([[0,0,0]])]*3,
            ('PROC', 12): [np.array([[i+1,0,0]]) for i in range(3)],
        }
    def select_atoms(self, query):
        # initial chain selection: 'segid PROA' or 'segid A'
        if query in ('segid PROA', 'segid A'):
            return self._chainA
        # G-G per-frame selection
        if query.startswith('segid PROA') and 'name CA' in query:
            # return group with 1 atom whose position for current frame
            frame = self._current_frame
            pos = self._positions[('PROA', 12)][frame]
            return FakeAtomGroup(residues=[], positions={'static': pos})
        if query.startswith('segid PROC') and 'name CA' in query:
            frame = self._current_frame
            pos = self._positions[('PROC', 12)][frame]
            return FakeAtomGroup(residues=[], positions={'static': pos})
        # everything else: empty group
        return FakeAtomGroup(residues=[])
    def __getattr__(self, attr):
        # allow trajectory read
        return getattr(self, attr)

@pytest.fixture(autouse=True)
def patch_universe(monkeypatch):
    # patch MDAnalysis.Universe to FakeUniverse
    monkeypatch.setattr(mda, 'Universe', FakeUniverse)
    # wrapped to set current_frame in FakeUniverse
    orig = FakeUniverse.__init__
    def init_and_track(self, psf, dcd):
        orig(self, psf, dcd)
        # wrap trajectory iteration to set frame
        new_traj = []
        for ts in self.trajectory:
            new_traj.append(ts)
        self.trajectory = new_traj
    monkeypatch.setattr(FakeUniverse, '__init__', init_and_track)
    # before each frame, set current_frame on universe
    def traj_iter(self):
        for ts in [FakeTS(i) for i in range(3)]:
            self._current_frame = ts.frame
            yield ts
    monkeypatch.setattr(FakeUniverse, 'trajectory', property(lambda self: list(FakeTS(i) for i in range(3))))
    # Need manual attribute on instance; override __iter__
    FakeUniverse.__iter__ = lambda self: (ts for ts in self.trajectory)
    return

def test_analyze_trajectory_control(tmp_path):
    # Create fake run directory
    run_dir = tmp_path / "run_control"
    run_dir.mkdir()
    # Write dummy psf and dcd so file exists
    (run_dir / "step5_input.psf").write_text("PSF")
    (run_dir / "MD_Aligned.dcd").write_text("DCD")
    # Call analyze_trajectory
    dist_ac, dist_bd, com_distances, time_points, system_dir, is_control = analyze_trajectory(
        str(run_dir), None, None
    )
    # For 3 frames, dist_ac should be [1,2,3]
    assert np.allclose(dist_ac, [1.0, 2.0, 3.0])
    # dist_bd should be all nan since 'B'/'D' not defined
    assert np.all(np.isnan(dist_bd))
    # com_distances should be None for control
    assert com_distances is None
    # time_points should be array of length 3
    assert len(time_points) == 3
    # is_control_system True
    assert is_control
