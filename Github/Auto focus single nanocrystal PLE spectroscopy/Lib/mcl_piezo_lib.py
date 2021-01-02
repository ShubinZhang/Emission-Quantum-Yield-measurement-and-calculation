#	Simple library to control Mad City Labs piezo stages
#  
# 	Original code by Yurii Morozov



verbose = False
from ctypes import *
import atexit
from time import sleep
import numpy as np

class madpiezo():
	def __init__(self):
		#self.madlib = cdll.LoadLibrary("C:\\Users\\Kuno Lab 4\\uscan\\ple_scan\\piezo\\Madlib.dll")
		self.madlib = cdll.LoadLibrary("Madlib.dll")
		self.handler = self.mcl_start()
		atexit.register(self.mcl_close)
	def mcl_start(self):
		"""
		Requests control of a single Mad City Labs Nano-Drive.

		Return Value:
			Returns a valid handle or returns 0 to indicate failure.
		"""
		mcl_init_handle = self.madlib['MCL_InitHandle']
		 
		mcl_init_handle.restype = c_int
		handler = mcl_init_handle()
		if(handler==0):
			print "MCL init error"
			return
		if verbose: print "handler = ", handler
		return 	handler
	def mcl_read(self,axis_number):
		"""
		Read the current position of the specified axis.
	
		Parameters:
			axis [IN] Which axis to move. (X=1,Y=2,Z=3,AUX=4)
			handle [IN] Specifies which Nano-Drive to communicate with.
		Return Value:
			Returns a position value or the appropriate error code.
		"""
		mcl_single_read_n = self.madlib['MCL_SingleReadN']
		mcl_single_read_n.restype = c_double
		return  mcl_single_read_n(c_uint(axis_number), c_int(self.handler))
	def mcl_write(self,position, axis_number):
		"""
		Commands the Nano-Drive to move the specified axis to a position.
	
		Parameters:
			position [IN] Commanded position in microns.
			axis [IN] Which axis to move. (X=1,Y=2,Z=3,AUX=4)
			handle [IN] Specifies which Nano-Drive to communicate with.
		Return Value:
			Returns MCL_SUCCESS or the appropriate error code.
		"""
		mcl_single_write_n = self.madlib['MCL_SingleWriteN']
		mcl_single_write_n.restype = c_int
		error_code = mcl_single_write_n(c_double(position), c_uint(axis_number), c_int(self.handler))
		
		if(error_code !=0):
			print "MCL write error = ", error_code
		return error_code
	def goxy(self,x_position,y_position):
		self.mcl_write(x_position,1)
		self.mcl_write(y_position,2)
	def goz(self,z_position):
		self.mcl_write(z_position,3)
	def get_position(self):
		return self.mcl_read(1), self.mcl_read(2), self.mcl_read(3)
	def mcl_close(self):
		"""
		Releases control of all Nano-Drives controlled by this instance of the DLL.
		"""
		mcl_release_all = self.madlib['MCL_ReleaseAllHandles']
		mcl_release_all()
if __name__ == "__main__":
	len_x = 64  # edit here
	len_y = 64  # edit here 
	x1, x2 = 0.,16. # edit here
	y1, y2 = 0., 16. # edit here
	x_pattern, y_pattern = np.meshgrid(np.linspace(x1, x2, len_x), np.linspace(y1, y2, len_y))
	scan_shape = np.shape(x_pattern)
	piezo = madpiezo()
	piezo.goxy(0.,0.)
	piezo.goz(50.)
	print piezo.get_position()
	
