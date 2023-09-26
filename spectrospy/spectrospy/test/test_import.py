

def test_dir_utils_materials():
    import spectrospy.misc.material
    d = dir(spectrospy.utils.material)
    assert d == [
        'atomic_to_weight',
        'density_of_mixture',
        'elements',
        'mass_absorption_coefficient',
        'mass_absorption_mixture',
        'weight_to_atomic',
        ]
