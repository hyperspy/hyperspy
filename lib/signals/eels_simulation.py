#import Signal
#
#class SimulationSignal(Signal):
#    def add_poissonian_noise(self):
#        '''Add Poissonian noise to the SI'''
#        self.__new_cube(np.random.poisson(self.data_cube).astype('float64'), 
#        'poissonian noise')
#        self._replot()
#
#    def add_gaussian_noise(self, std):
#        '''Add Gaussian noise to the SI
#        Parameters
#        ----------
#        std : float
#        
#        See also
#        --------
#        Spectrum.simulate
#        '''
#        self.__new_cube(np.random.normal(self.data_cube,std), 'gaussian_noise')
#        self._replot()
#        
#    def gaussian_filter(self, FWHM):
#        '''Applies a Gaussian filter in the energy dimension.
#        
#        Parameters
#        ----------
#        FWHM : float
#
#        See also
#        --------
#        Spectrum.simulate
#        '''
#        if FWHM > 0:
#            self.data_cube = gaussian_filter1d(self.data_cube, axis = 0, 
#            sigma = FWHM/2.35482)
#
#            
#    def add_energy_instability(self, std):
#        '''Introduce random energy instability
#        
#        Parameters
#        ----------
#        std : float
#            std in energy units of the energy instability.
#        See also
#        --------
#        Spectrum.simulate
#        '''
#        if abs(std) > 0:
#            delta_map = np.random.normal(
#            size = (self.xdimension, self.ydimension), 
#            scale = abs(std))
#        else:
#            delta_map = np.zeros((self.xdimension, 
#                    self.ydimension))
#        for edge in self.edges:
#            edge.delta.map = delta_map
#            edge.delta.already_set_map = np.ones((self.xdimension, 
#            self.ydimension), dtype = 'Bool')
#        return delta_map
#    
#    def create_data_cube(self):
#        '''Generate an empty data_cube from the dimension parameters
#        
#        The parameters self.energydimension, self.xdimension and 
#        self.ydimension will be used to generate an empty data_cube.
#        
#        See also
#        --------
#        Spectrum.simulate
#        '''
#        self.data_cube = np.zeros((self.energydimension, self.xdimension, 
#        self.ydimension))
#        self.get_dimensions_from_cube()
#        self.updateenergy_axis()
#        
#        
#    def simulate(self, maps = None, energy_instability = 0, 
#    min_intensity = 0., max_intensity = 1.):
#        '''Create a simulated SI.
#        
#        If an image is provided, it will use each RGB color channel as the 
#        intensity map of each three elements that must be previously defined as 
#        a set in self.elements. Otherwise it will create a random map for each 
#        element defined.
#        
#        Parameters:
#        -----------
#        maps : list/tuple of arrays
#            A list with as many arrays as elements are defined.
#        energy_instability : float
#            standard deviation in energy units of the energy instability.
#        min_intensity : float
#            minimum edge intensity
#        max_intensity : float
#            maximum edge intensity
#            
#        Returns:
#        --------
#        
#        If energy_instability != 0 it returns the energy shift map
#        '''
#        if maps is not None:
#            self.xdimension = maps[0].shape[0]
#            self.ydimension = maps[0].shape[1]
#            self.xscale = 1.
#            self.yscale = 1.
#            i = 0
#            if energy_instability > 0:
#                delta_map = np.random.normal(np.zeros((self.xdimension, 
#                self.ydimension)), energy_instability)
#            for edge in self.edges:
#                edge.fs_state = False
#                if not edge.intensity.twin:
#                    edge.intensity.map = maps[i]
#                    edge.intensity.already_set_map = np.ones((
#                    self.xdimension, self.ydimension), dtype = 'Bool')
#                    i += 1
#            if energy_instability != 0:
#                instability_map = self.add_energy_instability(energy_instability)
#            for edge in self.edges:
#                edge.charge_value_from_map(0,0)
#            self.create_data_cube()
#            self.model = Model(self, auto_background=False)
#            self.model.charge()
#            self.model.generate_cube()
#            self.data_cube = self.model.model_cube
#            self.type = 'simulation'
#        else:
#            print "No image defined. Producing a gaussian mixture image of the \
#            elements"
#            i = 0
#            if energy_instability:
#                delta_map = np.random.normal(np.zeros((self.xdimension, 
#                self.ydimension)), energy_instability)
#                print delta_map.shape
#            size = self.xdimension * self.ydimension
#            for edge in self.edges:
#                edge.fs_state = False
#                if not edge.intensity.twin:
#                    edge.intensity.map = np.random.uniform(0, max_intensity, 
#                    size).reshape(self.xdimension, self.ydimension)
#                    edge.intensity.already_set_map = np.ones((self.xdimension, 
#                    self.ydimension), dtype = 'Bool')
#                    if energy_instability:
#                        edge.delta.map = delta_map
#                        edge.delta.already_set_map = np.ones((self.xdimension, 
#                        self.ydimension), dtype = 'Bool')
#                    i += 1
#            self.create_data_cube()
#            self.model = Model(self, auto_background=False)
#            self.model.generate_cube()
#            self.data_cube = self.model.model_cube
#            self.type = 'simulation'
#        if energy_instability != 0:
#            return instability_map