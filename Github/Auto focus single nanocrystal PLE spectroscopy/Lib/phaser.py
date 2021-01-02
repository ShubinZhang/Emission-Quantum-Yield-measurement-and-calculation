#This is a library for communicating with the lasers
from time import sleep
verbose = True
import serial
import atexit, string, time
debug = True
import numpy as np
class obis():
	def __init__(self, port):
		self.laser_is_on = False
		self.port_is_open = False
		self.wlength = laser_wlength
		self.ser = serial.Serial()
		self.ser.baudrate = 9600
		self.ser.port = "COM" + str(port)
		self.ser.timeout = 1
		if self.ser.isOpen():
			self.ser.close()
			sleep(0.3)
		else:
			try:
				self.ser.open()
			except serial.SerialException:
				print "port is already open..."	
		if self.ser.isOpen():
			self.port_is_open = True
		self.ser.flushInput()
		self.ser.write("SOUR:AM:STATE OFF\r\n") #turn Off the laser
		if verbose: print self.ser.readline()
		atexit.register(self.close)
	def start(self):
		"""
		try:
			self.ser.write("SOUR:AM:INT CWP\r\n") #set operation mode to internal CW Power
		except serial.SerialException:
			self.ser.close()
			self.ser.open()
			self.ser.write("SOUR:AM:INT CWP\r\n") #set operation mode to internal CW Power
		print self.ser.readline()
		print "SOUR:POW:LEV:IMM:AMPL"+" "+str(self.power/1000.)+"\r\n"
		self.ser.write("SOUR:POW:LEV:IMM:AMPL"+" "+str(self.power/1000.)+"\r\n") # set power level
		print self.ser.readline()
		"""
		self.ser.write("SOUR:AM:STATE ON\r\n")#turn on laser
		if verbose: print self.ser.readline()
		self.laser_is_on = True
		if verbose: print self.ser.readline()
	def stop(self):
		if self.laser_is_on:
			self.laser_is_on = False
			self.ser.write("SOUR:AM:STATE OFF\r\n") #turn Off the laser
			if verbose: print self.ser.readline()
		else:
			print "Laser connected to port ", str(self.ser.port), " is already stopped..."
	def set_analog_power(self):
		self.ser.write("SOUR:AM:EXT ANAL\r\n") #set operation mode to external analog power modulation
	def set_cw_power(self):
		self.ser.write("SOUR:AM:INT CWP\r\n") #set operation mode to internal CW Power
		if verbose: print self.ser.readline()
	def set_power(self, power):
		self.power = float(power)
		#set laser power
		self.ser.write("SOUR1:POW:LEV:IMM:AMPL "+str(self.power/1000.)+"\r\n") # set power level
		if verbose: print self.ser.readline()
		if verbose: print self.ser.readline()
		self.ser.write("SOUR1:POW:LEV:IMM:AMPL?"+"\r\n") # set power level
		print "ask laser power that was setted"
		if verbose: print self.ser.readline()
		if verbose: print self.ser.readline()
	def close(self):
		if self.port_is_open:
			#self.ser.write("SOUR:AM:STATE OFF\r\n") #turn Off the laser
			#turn off
			self.ser.close()
			if not(self.ser.isOpen()):
				self.port_is_open = False
			if verbose: print "Closing laser com port..."
			if verbose: print "Is Open?", self.ser.isOpen()
		else:
			print "Laser connected to port ", str(self.ser.port), " is already closed..."


if __name__ == "__main__":
	laser = obis("730")
	laser.start()
	laser.set_cw_power()
	print "setting power"
	laser.set_power(0)
	powers = np.arange(0.,30.,0.5)
	for cur_power in powers:
		print "cur_power = ", cur_power
		laser.set_power(cur_power)
		#sleep(0.05)
