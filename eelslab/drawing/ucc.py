#!/usr/bin/env python
# Copyright 2011 Michael Sarahan
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA

# Unit Cell Cropper
# Designed to facilitate locating and cropping unit cells from high-resolution
# images of crystalline lattices.

from enthought.traits.api \
    import HasTraits, Array, Int, Float, Range, Instance, on_trait_change, \
    Bool, Button, Property, Event, Tuple, Any, List
from enthought.traits.ui.api import View, Item, Group, HFlow, VGroup, Tabbed, \
    BooleanEditor, ButtonEditor, CancelButton, Handler, Action, Spring, \
    HGroup
from enthought.chaco.tools.api import PanTool, ZoomTool, RangeSelection, \
                                       RangeSelectionOverlay
from enthought.chaco.api import Plot, ArrayPlotData, jet, gray, \
    ColorBar, ColormappedSelectionOverlay, HPlotContainer, LinearMapper, \
    OverlayPlotContainer
from enthought.enable.api import ComponentEditor, KeySpec, Component
from enthought.chaco.tools.cursor_tool import CursorTool, BaseCursorTool
from enthought.traits.ui.key_bindings import KeyBinding, KeyBindings


from eelslab import cv_funcs
import numpy as np  

class OK_custom_handler(Handler):
    def close(self,info, is_ok):
        if is_ok:
            info.object.crop_cells_stack()
        return True      

key_bindings = KeyBindings(
    KeyBinding( binding1    = 'Left',
                description = 'Step left through images',
                method_name = 'decrease_img_idx' ),
    KeyBinding( binding1    = 'Right',
                description = 'Step right through images',
                method_name = 'increase_img_idx' ),
)

#(value=[Array(dtype=np.float32,value=np.zeros((64,64)))])
# (value=[Array(shape=(None,3), value=np.array(([[0,0,0]])))])
class TemplatePicker(HasTraits):
    template = Array
    CC = List
    peaks = List
    zero=Int(0)
    tmp_size = Range(low=2, high=512, value=64, cols=4)
    max_pos_x=Int(1023)
    max_pos_y=Int(1023)
    top = Range(low='zero',high='max_pos_x', value=20, cols=4)
    left = Range(low='zero',high='max_pos_y', value=20, cols=4)
    is_square = Bool
    img_plot = Instance(Plot)
    tmp_plot = Instance(Plot)
    findpeaks = Button
    peak_width = Range(low=2, high=200, value=10)
    tab_selected = Event
    ShowCC = Bool
    img_container = Instance(Component)
    container = Instance(Component)
    colorbar= Instance(Component)
    numpeaks_total = Int(0)
    numpeaks_img = Int(0)
    OK_custom=OK_custom_handler
    cbar_selection = Instance(RangeSelection)
    cbar_selected = Event
    thresh=Tuple(0.0,1.0)
    numfiles=Int(1)
    img_idx=Int(0)
    tmp_img_idx=Int(0)

    csr=Instance(BaseCursorTool)

    traits_view = View(
        HFlow(
            VGroup(
                Item("img_container",editor=ComponentEditor(), show_label=False),
                Group(
                    Spring(),
                    Item("ShowCC", editor=BooleanEditor(), label="Show cross correlation image")),
                label="Original image", show_border=True, trait_modified="tab_selected"
                ),
            VGroup(
                Group(
                    HGroup(
                        Item("left", label="Left coordinate", style="custom"),
                        Item("top", label="Top coordinate", style="custom"),
                        ),
                    Item("tmp_size", label="Template size", style="custom"),
                    Item("tmp_plot",editor=ComponentEditor(height=256, width=256), show_label=False, resizable=True),
                    label="Template", show_border=True),
                Group(
                    Item("peak_width", label="Peak width", style="custom"),
                    Group(
                        Spring(),
                        Item("findpeaks",editor=ButtonEditor(label="Find Peaks"),show_label=False),
                        Spring(),
                        ),
                    Group(
                        Spring(),
                        Item("numpeaks_img",label="Number of Cells selected (this image)",style='readonly'),
                        Item("numpeaks_total",label="Total",style='readonly'),                        
                        ),
                    label="Peak parameters", show_border=True),
                )
            ),
        buttons = [ Action(name='OK', enabled_when = 'numpeaks_total > 0' ),
            CancelButton ],
        title="Template Picker",
        handler=OK_custom, kind='livemodal',
        key_bindings = key_bindings,
        width=870, height=540)        

    def __init__(self, signal_instance):
        super(TemplatePicker, self).__init__()
        try:
            import cv
        except:
            print "OpenCV unavailable.  Can't do cross correlation without it.  Aborting."
            return None
        self.OK_custom=OK_custom_handler()
        self.sig=signal_instance
        if not hasattr(self.sig.mapped_parameters,"original_files"):
            self.sig.data=np.atleast_3d(self.sig.data)
            self.titles=[self.sig.mapped_parameters.name]
        else:
            self.numfiles=len(self.sig.mapped_parameters.original_files.keys())
            self.titles=self.sig.mapped_parameters.original_files.keys()
        tmp_plot_data=ArrayPlotData(imagedata=self.sig.data[self.top:self.top+self.tmp_size,self.left:self.left+self.tmp_size,self.img_idx])
        tmp_plot=Plot(tmp_plot_data,default_origin="top left")
        tmp_plot.img_plot("imagedata", colormap=jet)
        tmp_plot.aspect_ratio=1.0
        self.tmp_plot=tmp_plot
        self.tmp_plotdata=tmp_plot_data
        self.img_plotdata=ArrayPlotData(imagedata=self.sig.data[:,:,self.img_idx])
        self.img_container=self._image_plot_container()

        self.crop_sig=None

    def render_image(self):
        plot = Plot(self.img_plotdata,default_origin="top left")
        img=plot.img_plot("imagedata", colormap=gray)[0]
        plot.title="%s of %s: "%(self.img_idx+1,self.numfiles)+self.titles[self.img_idx]
        plot.aspect_ratio=float(self.sig.data.shape[1])/float(self.sig.data.shape[0])

        #if not self.ShowCC:
        csr = CursorTool(img, drag_button='left', color='white',
                         line_width=2.0)
        self.csr=csr
        csr.current_position=self.left, self.top
        img.overlays.append(csr)

        # attach the rectangle tool
        plot.tools.append(PanTool(plot,drag_button="right"))
        zoom = ZoomTool(plot, tool_mode="box", always_on=False, aspect_ratio=plot.aspect_ratio)
        plot.overlays.append(zoom)
        self.img_plot=plot
        return plot

    def render_scatplot(self):
        peakdata=ArrayPlotData()
        peakdata.set_data("index",self.peaks[self.img_idx][:,0])
        peakdata.set_data("value",self.peaks[self.img_idx][:,1])
        peakdata.set_data("color",self.peaks[self.img_idx][:,2])
        scatplot=Plot(peakdata,aspect_ratio=self.img_plot.aspect_ratio,default_origin="top left")
        scatplot.plot(("index", "value", "color"),
                      type="cmap_scatter",
                      name="my_plot",
                      color_mapper=jet,
                      marker = "circle",
                      fill_alpha = 0.5,
                      marker_size = 6,
                      )
        scatplot.x_grid.visible = False
        scatplot.y_grid.visible = False
        scatplot.range2d=self.img_plot.range2d
        self.scatplot=scatplot
        self.peakdata=peakdata
        return scatplot

    def _image_plot_container(self):
        plot = self.render_image()

        # Create a container to position the plot and the colorbar side-by-side
        self.container=OverlayPlotContainer()
        self.container.add(plot)
        self.img_container = HPlotContainer(use_backbuffer = False)
        self.img_container.add(self.container)
        self.img_container.bgcolor = "white"

        if self.numpeaks_img>0:
            scatplot = self.render_scatplot()
            self.container.add(scatplot)
            colorbar = self.draw_colorbar()
            self.img_container.add(colorbar)
        return self.img_container

    def draw_colorbar(self):
        scatplot=self.scatplot
        cmap_renderer = scatplot.plots["my_plot"][0]
        selection = ColormappedSelectionOverlay(cmap_renderer, fade_alpha=0.35, 
                                                selection_type="mask")
        cmap_renderer.overlays.append(selection)

        # Create the colorbar, handing in the appropriate range and colormap
        colorbar = self.create_colorbar(scatplot.color_mapper)
        colorbar.plot = cmap_renderer
        colorbar.padding_top = scatplot.padding_top
        colorbar.padding_bottom = scatplot.padding_bottom
        self.colorbar=colorbar
        return colorbar

    def create_colorbar(self,colormap):
        colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                        color_mapper=colormap,
                        orientation='v',
                        resizable='v',
                        width=30,
                        padding=20)
        colorbar_selection=RangeSelection(component=colorbar)
        colorbar.tools.append(colorbar_selection)
        colorbar.overlays.append(RangeSelectionOverlay(component=colorbar,
                                                   border_color="white",
                                                   alpha=0.8,
                                                   fill_color="lightgray",
                                                   metadata_name='selections'))
        colorbar_selection.selection=self.thresh
        colorbar.selections=self.thresh
        print colorbar_selection.selection
        self.cbar_selection=colorbar_selection
        return colorbar

    @on_trait_change('ShowCC')
    def toggle_cc_view(self):
        if self.ShowCC:
            self.CC[self.img_idx] = cv_funcs.xcorr(self.sig.data[self.top:self.top+self.tmp_size,
                                                   self.left:self.left+self.tmp_size,self.img_idx],
                                     self.sig.data)
            self.img_plotdata.set_data("imagedata",self.CC[self.img_idx])
        else:
            self.img_plotdata.set_data("imagedata",self.sig.data[:,:,self.img_idx])
        self.redraw_plots()

    @on_trait_change("img_idx")
    def update_img_depth(self):
        self.img_plotdata.set_data("imagedata",self.sig.data[:,:,self.img_idx])
        self.img_plot.title="%s of %s: "%(self.img_idx+1,self.numfiles)+self.titles[self.img_idx]
        self.redraw_plots()

    @on_trait_change("peaks")
    def redraw_plots(self):
        oldplot=self.img_plot
        self.container.remove(oldplot)
        newplot=self.render_image()
        self.container.add(newplot)
        self.img_plot=newplot

        try:
            # if these haven't been created before, this will fail.  wrap in try to prevent that.
            oldscat=self.scatplot
            self.container.remove(oldscat)
            oldcolorbar = self.colorbar
            self.img_container.remove(oldcolorbar)
        except:
            pass

        if self.numpeaks_img>0:
            newscat=self.render_scatplot()
            self.container.add(newscat)
            self.scatplot=newscat
            colorbar = self.draw_colorbar()
            self.img_container.add(colorbar)
            self.colorbar=colorbar

        self.container.request_redraw()
        self.img_container.request_redraw()        

    @on_trait_change('tmp_size')
    def update_max_pos(self):
        max_pos_x=self.sig.data.shape[0]-self.tmp_size-1
        if self.left>max_pos_x: self.left=max_pos_x
        self.max_pos_x=max_pos_x
        max_pos_y=self.sig.data.shape[1]-self.tmp_size-1
        if self.top>max_pos_y: self.top=max_pos_y
        self.max_pos_y=max_pos_y
        return

    def increase_img_idx(self,info):
        if self.img_idx==(self.numfiles-1):
            self.img_idx=0
        else:
            self.img_idx+=1

    def decrease_img_idx(self,info):
        if self.img_idx==0:
            self.img_idx=self.numfiles-1
        else:
            self.img_idx-=1

    @on_trait_change('left, top')
    def update_csr_position(self):
        self.csr.current_position=self.left,self.top

    @on_trait_change('csr:current_position')
    def update_top_left(self):
        self.left,self.top=self.csr.current_position
        
    @on_trait_change('left, top, tmp_size')
    def update_tmp_plot(self):
        self.tmp_plotdata.set_data("imagedata", 
                                   self.sig.data[self.top:self.top+self.tmp_size,self.left:self.left+self.tmp_size,self.img_idx])
        grid_data_source = self.tmp_plot.range2d.sources[0]
        grid_data_source.set_data(np.arange(self.tmp_size), np.arange(self.tmp_size))
        self.tmp_img_idx=self.img_idx
        return

    @on_trait_change('left, top, tmp_size')
    def update_CC(self):
        if self.ShowCC:
            self.CC[self.img_idx] = cv_funcs.xcorr(self.sig.data[self.top:self.top+self.tmp_size,
                                                   self.left:self.left+self.tmp_size,self.tmp_img_idx],
                                     self.sig.data[:,:,self.img_idx])
            self.img_plotdata.set_data("imagedata",self.CC[self.img_idx])
            grid_data_source = self.img_plot.range2d.sources[0]
            grid_data_source.set_data(np.arange(self.CC[self.img_idx].shape[1]), 
                                      np.arange(self.CC[self.img_idx].shape[0]))
        if self.numpeaks_total>0:
            self.peaks=[np.array([[0,0,-1]])]

    @on_trait_change('peaks,cbar_selection:selection_completed')
    def calc_numpeaks(self):
        try:
            thresh=self.cbar_selection.selection
        except:
            thresh=[]
        if thresh==[]:
            thresh=(0,1)
        self.thresh=thresh
        self.numpeaks_total=np.sum([np.sum(np.ma.masked_inside(self.peaks[i][:,2],thresh[0],thresh[1]).mask) for i in xrange(len(self.peaks))])
        self.numpeaks_img=np.sum(np.ma.masked_inside(self.peaks[self.img_idx][:,2],thresh[0],thresh[1]).mask)
        #except:
        #    pass

    @on_trait_change('findpeaks')
    def locate_peaks(self):
        from eelslab import peak_char as pc
        peaks=[]
        for idx in xrange(self.numfiles):
            try:
                self.CC[idx] = cv_funcs.xcorr(self.sig.data[self.top:self.top+self.tmp_size,
                                               self.left:self.left+self.tmp_size,self.tmp_img_idx],
                                 self.sig.data[:,:,idx])
            except:
                self.CC.append(cv_funcs.xcorr(self.sig.data[self.top:self.top+self.tmp_size,
                                               self.left:self.left+self.tmp_size,self.tmp_img_idx],
                                 self.sig.data[:,:,idx]))
            pks=pc.two_dim_findpeaks(self.CC[idx]*255, peak_width=self.peak_width, medfilt_radius=None)
            pks[:,2]=pks[:,2]/255
            peaks.append(pks)
        self.peaks=peaks
        
    def mask_peaks(self,idx):
        thresh=self.cbar_selection.selection
        if thresh==[]:
            thresh=(0,1)
        mpeaks=np.ma.asarray(self.peaks[idx])
        mpeaks[:,2]=np.ma.masked_outside(mpeaks[:,2],thresh[0],thresh[1])
        return mpeaks

    def crop_cells_stack(self):
        from eelslab.signals.aggregate import AggregateCells
        if self.numfiles==1:
            self.crop_sig=self.crop_cells()
            return
        else:
            crop_agg=[]
            for idx in xrange(self.numfiles):
                crop_agg.append(self.crop_cells(idx))
                crop_agg[idx].mapped_parameters.print_items()
            self.crop_sig=AggregateCells(*crop_agg)
            return

    def crop_cells(self,idx=0):
        print "cropping cells..."
        from eelslab.signals.image import Image
        # filter the peaks that are outside the selected threshold
        peaks=np.ma.compress_rows(self.mask_peaks(idx))
        tmp_sz=self.tmp_size
        data=np.zeros((tmp_sz,tmp_sz,peaks.shape[0]))
        for i in xrange(peaks.shape[0]):
            # crop the cells from the given locations
            data[:,:,i]=self.sig.data[peaks[i,1]:peaks[i,1]+tmp_sz,peaks[i,0]:peaks[i,0]+tmp_sz,idx]
            crop_sig=Image({'data':data,
                            'mapped_parameters':{
                                'name':'Cropped cells from %s'%self.titles[idx],
                                'data_type':'Image',
                                'locations':peaks,
                                'parent':self.titles[idx],
                                }
                         })
        return crop_sig
        # attach a class member that has the locations from which the images were cropped
        print "Complete.  "

if __name__=="__main__":
    import sys
    from eelslab.EELSlab import *
    sig=load("005nm_0.7500_frac.png")
    #if sys.flags.interactive !=1:
    #    app.exec_()
    #pyqt_template_picker(sig)
    
    TemplatePicker(sig)
