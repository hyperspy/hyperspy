#from eelslab import Signal
#
#
#class PESSignal(Signal):
#    def remove_Shirley_background(self, max_iter = 10, eps = 1e-6):
#        """Remove the inelastic background of photoemission SI by the shirley 
#        iterative method.
#        
#        Parameters
#        ----------
#        max_iter : int
#            maximum number of iterations
#        eps : float
#            convergence limit
#        """
#        bg_list = []
#        iter = 0
#        s = self.data_cube.copy()
#        a = s[:3,:,:].mean()
#        b = s[-3:,:,:].mean()
#        B = b * 0.9 * np.ones(s.shape)
#        bg_list.append(B)
#        mean_epsilon = 10*eps
#        integral = None
#        old_integral = None
#        while  (mean_epsilon > eps) and (iter < max_iter):
#            if integral is not None:
#                old_integral = integral
#            sb = s - B
#            integral = np.cumsum(
#            sb[::-1,:,:], axis = 0)[::-1, :, :] * self.energyscale
#            B = (a-b)*integral/integral[0,:,:] + b
#            bg_list.append(B)
#            if old_integral is not None:
#                epsilon = np.abs(integral[0,:,:] - old_integral[0,:,:])
#                mean_epsilon = epsilon.mean()
#            print "iter: %s\t mean epsilon: %s" % (iter, mean_epsilon)
#            iter += 1
#        self.data_cube = sb
#    return epsilon, bg_list