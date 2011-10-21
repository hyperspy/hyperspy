//
// Copyright 2011 Luiz Fernando Zagonel
// ImportRPL is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.

// ImportRPL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with Hyperspy; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
// USA

//  This script is designed to import Ripple files into Digital Micrograph. 
//  It is used to easy data transit between DigitalMicrograph and Hyperspy. 
//  The script will ask for 2 files: 
//          1- the riple file with the data  format and calibrations 
//          2- the data itself in raw format.
//          IF a file with the same name and path as the riple file exits
//          with raw or bin extension it is opened directly without prompting

// Last modified 20/10/2011, Mike Sarahan
// Spelling fixes, added conversion to SI if eV is in units
// Made 2D images nicer to look at (no longer requires slice tool)

Object file_stream
Image img
String filenameRPL, filenameRAW
Number size_x, size_y, size_z, pixel_size
realnumber   scale_x, scale_y,scale_z, origin_x, origin_y, origin_z
string units_x, units_y, units_z
Number file_RPL, file_RAW
Number index
string textline,value,data_type_value,recorded_by_value
string width_name_value, height_name_value, depth_name_value
string imagename

size_x = size_y = size_z = 1

//Prompt the user to locate the Ripple file
If (!OpenDialog(filenameRPL)) Exit(0)

//opens the riple file
file_RPL = OpenFileForReading(filenameRPL)

//Read ripple file line by line
While(ReadFileLine(file_RPL, textline))
{
//Result(textline + "\n") //display each line read

//Look for Keywords to read format and calibration
if(textline.find("width")!=-1)
if(textline.mid(5,1)!="-")
       {
       value = textline.right(textline.len() - 5 )
       size_x =  value.val()
       }

if(textline.find("width-scale")!=-1)
       { 
       value = textline.right(textline.len() - 11)
       scale_x =  value.val()
       }
           
if(textline.find("width-origin")!=-1)
       { 
       value = textline.right(textline.len() - 12)
       origin_x =  value.val()
       }

if(textline.find("width-name")!=-1)
       { 
      width_name_value = textline.right(textline.len() - 11)
      width_name_value = width_name_value.left(width_name_value.len()-2)
       }

if(textline.find("width-units")!=-1)
       { 
      units_x = textline.right(textline.len() - 12)
      units_x = units_x.left(units_x.len()-2)
       }

if(textline.find("height")!=-1)
if(textline.mid(6,1)!="-")
       {
       value = textline.right(textline.len() - 6)
       size_y =  value.val()
       }
       
if(textline.find("height-scale")!=-1)
       { 
       value = textline.right(textline.len() - 12)
       scale_y =  value.val()
       }
           
if(textline.find("height-origin")!=-1)
       { 
       value = textline.right(textline.len() - 13)
       origin_y =  value.val()
       }
       
if(textline.find("height-name")!=-1)
       { 
      height_name_value = textline.right(textline.len() - 12)
      height_name_value = height_name_value.left(height_name_value.len()-2)
       }

if(textline.find("height-units")!=-1)
       { 
      units_y = textline.right(textline.len() - 13)
      units_y = units_y.left(units_y.len()-2)
       }
        
if(textline.find("depth")!=-1)
if(textline.mid(5,1)!="-")
       { 
       value = textline.right(textline.len() - 5)
       size_z =  value.val()
       }
       
if(textline.find("depth-origin")!=-1)
       { 
       value = textline.right(textline.len() - 12)
       origin_z =  value.val()
       }
       
if(textline.find("depth-scale")!=-1)
       { 
       value = textline.right(textline.len() - 11)
       scale_z =  value.val()
       }
       
if(textline.find("depth-units")!=-1)
       { 
      depth_name_value = textline.right(textline.len() - 11)
      depth_name_value = depth_name_value.left(depth_name_value.len()-2)
       }

if(textline.find("depth-units")!=-1)
       { 
      units_z = textline.right(textline.len() - 12)
      units_z = units_z.left(units_z.len()-2)
       }        

if(textline.find("data-length")!=-1)
      {
      value = textline.right(textline.len() - 11)      
      pixel_size =  value.val()
      }
      
if(textline.find("data-type")!=-1)
      {      
      data_type_value = textline.right(textline.len() - 10)
      data_type_value = data_type_value.left(data_type_value.len()-2)
      }
      
      

if(textline.find("record-by")!=-1)
      {      
      recorded_by_value = textline.right(textline.len() - 10)
      recorded_by_value = recorded_by_value.left(recorded_by_value.len()-2)
      }

      
}
//Riple file reading is finished
CloseFile(file_RPL)


if(data_type_value=="signed")
if(pixel_size>4)
{

//Give some info on the information obtained from the riple file. 
Result( "Data dimensions:" + size_x + " x " + size_y + " x " +size_z + " x " + pixel_size + " \n" )  
Result( "Scales:" + scale_x + " x " + scale_y + " x " + scale_z + " \n" )  
Result( "Origins:" + origin_x + " x " + origin_y + " x " + origin_z +" \n" )
Result( "Units:" + units_x + " x " + units_y + " x " + units_z +" \n" ) 
Result("Data type is "+ data_type_value + ".\n")
Result( "This script can not open data with data type integer and pixel size higher than 4. \n")
Exit(0)
}


// Check for a file with same name of riple but with raw extension
file_RAW = -1
filenameRAW = filenameRPL.left(filenameRPL.len() - 3 )
filenameRAW = filenameRAW+"raw"

if(!DoesFileExist(filenameRAW)) // if no such file is found
{
// Check for a file with same name of riple but with bin extension
filenameRAW = filenameRPL.left(filenameRPL.len() - 3 )
filenameRAW = filenameRAW+"bin"

if(!DoesFileExist(filenameRAW))
// if no such file is found, prompts the user for a raw or bin file
{
OkDialog("Raw or Bin file not found in path. Please select file manually." ) 
If (!OpenDialog(filenameRAW)) Exit(0)
}

}

//opens the raw (or bin or file given by user) file
file_RAW = OpenFileForReading(filenameRAW)
file_stream = NewStreamFromFileReference(file_RAW, 1)

imagename = PathExtractFileName(filenameRAW,0)


if(recorded_by_value=="vector")
{
if(data_type_value=="signed"){
        if (size_y>1){
        img := IntegerImage(imagename, pixel_size, 1, size_x, size_z, size_y)
        }
        else {
        img := IntegerImage(imagename, pixel_size, 1, size_x, size_z)
        }
        }
if(data_type_value=="float") {
        if (size_y>1){  
        img := RealImage(imagename, pixel_size, size_x, size_z, size_y)
        }
        else {
        img := RealImage(imagename, pixel_size, size_x, size_z)
        }
        }
if(data_type_value=="unsigned") {
        if (size_y>1){  
        img := IntegerImage(imagename, pixel_size, 0, size_x, size_z, size_y)
        }
        else {
        img := IntegerImage(imagename, pixel_size, 0, size_x, size_z)
        }
        }
if(!ImageIsValid(img))
{
        if (size_y>1) {  
        img := RealImage(imagename, pixel_size, size_x, size_z, size_y)
        }
        else {
        img := RealImage(imagename, pixel_size, size_z, size_x)
        }
}

  ImageReadImageDataFromStream(img, file_stream, 0)

  img.RotateLeft()
  img.flipvertical()

	img.ImageSetDimensionOrigin(0, origin_z )  
	img.ImageSetDimensionScale( 0, scale_z )  
	img.ImageSetDimensionUnitString(0, units_z )

if(size_x>1)
{
img.ImageSetDimensionOrigin(1, origin_x )  
img.ImageSetDimensionScale( 1, scale_x )  
img.ImageSetDimensionUnitString(1, units_x )  
}

if(size_y>1)
{
img.ImageSetDimensionOrigin(2, origin_y )  
img.ImageSetDimensionScale( 2, scale_y )  
img.ImageSetDimensionUnitString(2, units_y )  
}
}

else {
        
//Create an image depending on the data-type
if(data_type_value=="signed"){
        if (size_y>1){
        img := IntegerImage(imagename, pixel_size, 1, size_x, size_y, size_z)
        }
        else{
        img := IntegerImage(imagename, pixel_size, 1, size_x, size_y)
        }
        }

if(data_type_value=="float") {
        if (size_y>1){
        img := RealImage(imagename, pixel_size, size_x, size_y, size_z)
        }
        else {
        img := RealImage(imagename, pixel_size, size_x, size_y)
        }
        }

if(data_type_value=="unsigned") {
        if (size_y>1){
        img := IntegerImage(imagename, pixel_size, 0, size_x, size_y, size_z)
        }
        else {
        img := IntegerImage(imagename, pixel_size, 0, size_x, size_y)
        }
        }
//if data-type was not given or recognized
if(!ImageIsValid(img))
{
result(" Unrecognized data-type value. Data-type formats are: signed, unsigned and float. Using float by default."+"\n")
        if (size_y>1){
        img := RealImage(imagename, pixel_size, size_x, size_y, size_z)
        }
        else {
        img := RealImage(imagename, pixel_size, size_x, size_y)
        }
}

//if the data is recorded by vector, create another image to change the data order

ImageReadImageDataFromStream(img, file_stream, 0)

//Add scale calibration and format to the image
if(size_x>1)
{
img.ImageSetDimensionOrigin(0, origin_x )  
img.ImageSetDimensionScale( 0, scale_x )  
img.ImageSetDimensionUnitString(0, units_x )  
}

if(size_y>1)
{
img.ImageSetDimensionOrigin(1, origin_y )  
img.ImageSetDimensionScale( 1, scale_y )  
img.ImageSetDimensionUnitString(1, units_y )  
}

if(size_z>1)
{
	img.ImageSetDimensionOrigin(2, origin_z )  
	img.ImageSetDimensionScale( 2, scale_z )  
	img.ImageSetDimensionUnitString(2, units_z )   
}

}
//data reading if finished
CloseFile(file_RAW)

if (units_x=="eV" || units_y=="eV" || units_z=="eV") {
    img.setstringnote("Meta Data:Format","Spectrum image")
    img.setstringnote("Meta Data:Signal","EELS")
    }

img.ShowImage()

//if(size_x==1) //if the data is a collection of spectra, display in raster mode
//setDisplayType(img,4)

UpdateImage (Img)
