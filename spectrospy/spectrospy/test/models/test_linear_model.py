@lazifyTestClass
class TestLinearEELSFitting:

    def setup_method(self, method):
        ll = hs.datasets.artificial_data.get_low_loss_eels_signal()
        cl = hs.datasets.artificial_data.get_core_loss_eels_signal()
        cl.add_elements(('Mn',))
        m = cl.create_model(auto_background=False)
        m[0].onset_energy.value = 673.
        m_convolved = cl.create_model(auto_background=False, ll=ll)
        m_convolved[0].onset_energy.value = 673.
        self.ll, self.cl = ll, cl
        self.m, self.m_convolved = m, m_convolved

    def test_convolved_and_std_error(self):
        m = self.m_convolved
        m.fit(optimizer='lstsq')
        linear = m.as_signal()
        std_linear = m.p_std
        m.fit(optimizer='lm')
        lm = m.as_signal()
        std_lm = m.p_std
        diff = linear - lm
        np.testing.assert_allclose(diff.data.sum(), 0.0, atol=1E-6)
        np.testing.assert_allclose(std_linear, std_lm)

    def test_nonconvolved(self):
        m = self.m
        m.fit(optimizer='lstsq')
        linear = m.as_signal()
        m.fit(optimizer='lm')
        lm = m.as_signal()
        diff = linear - lm
        np.testing.assert_allclose(diff.data.sum(), 0.0, atol=1E-6)
