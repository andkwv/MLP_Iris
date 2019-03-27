#NO LOOP NO FUNCTION NO NOTHING HERE WE GO STRAIGHT TO THE POINT FUCKALL
import csv
import random as rd
import math as mt
import numpy as np
# libraries for the graph only
import matplotlib.pyplot as plt



class IrisData():
	def __init__(self, filename, block_num):
		with open(filename, "r") as f_input:
			csv_input = csv.reader(f_input)
			self.details = list(csv_input)
			# self.details = np.array(detail)

			#create 5 lists inside the block list to separate into 5 blocks
			self.blocks = [[] for _ in range(5)] 

			#split the data into two blocks
			self.validator = []
			self.trainer = []

			self.validate_block = block_num

			#we proceed to restructure the data, and put the data inside the blocks
			self.restructure()

	# FOR K-FOLD INTO 5 ROWS
	# def restructure(self):
	# 	detail_copy = self.details

	# 	for row in range(150):
	# 		block_num = row % 5
	# 		self.blocks[block_num].append(detail_copy[row])
	# 	#then we proceed to randomize every block
	# 	for block in range(5):
	# 		self.randomize(block)

	def restructure(self):
		detail_copy = self.details
		
		iris_s = detail_copy[:50]
		iris_ve = detail_copy[50:100]
		iris_vi = detail_copy[100:150]

		rd.shuffle(iris_s)
		rd.shuffle(iris_ve)
		rd.shuffle(iris_vi)

		self.split(iris_s, iris_ve, iris_vi)

	def split(self, setosa, versicolor, virginica):
		#we take 10 of each types and put them in the validator block

		for i in range(10):
			self.validator.append(setosa.pop(i))
			self.validator.append(versicolor.pop(i))
			self.validator.append(virginica.pop(i))

		self.trainer = setosa + versicolor + virginica
		rd.shuffle(self.trainer)
		# rd.shuffle(self.validator)


	# function to randomize one of the blocks
	def randomize(self, num_block):
		#randomize the list
		rd.shuffle(self.blocks[num_block])

	def get_col_row(self, block, row, col):
		if(block == 1):
			return self.get_col_row_trainer(row, col)
		else:
			return self.get_col_row_validator(row, col)

	def get_col_row_trainer(self, row, col):
		return self.trainer[row][col]
		# Python index starts from 0 so we have to substract by 1

	def get_col_row_validator(self, row, col):
		return self.validator[row][col]


	def get_splitData(self):
		print("\nThis is the Validation Block")

		for i in range(30):
			print("%s. %s" % ((i+1),self.validator[i]))

		print("\nThis is the Training Block")

		for j in range(120):
			print("%s. %s" % ((j+1),self.trainer[j]))


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX  CALCULATOR   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX #

class Calculation():
	def __init__(self, data_object):
		self.data = data_object

	# the function to retrieve values from the csv file
	def getSepalPetal(self, which_block, row):

		sepal_length = float(self.data.get_col_row(which_block, row, 0))
		sepal_width = float(self.data.get_col_row(which_block, row, 1))
		petal_length = float(self.data.get_col_row(which_block, row, 2))
		petal_width = float(self.data.get_col_row(which_block, row,3))

		y1 = float(self.data.get_col_row(which_block, row,5))
		y2 = float(self.data.get_col_row(which_block, row,6))

		# assigned to a list to simplify process 
		SepalPetal_list = [sepal_length, sepal_width, petal_length, petal_width, y1, y2]
		return SepalPetal_list
		# we simply return the list

	# the values will be pre calculated before calling the target calculation function
	def target(self, t1, t2, t3, t4, b):
		result = t1 + t2 + t4 + b
		return result

	# we calculate the sigmoid simply using the formula
	def output(self, targetx):
		exp = mt.exp(-targetx)
		output = 1 / (1 + exp)
		return output

	# we normalize, if it is below 0.5 then we will predict it to be 1
	def normalPredict(self, output_var):
		if(output_var < 0.5):
			return 0
		else:
			return 1

	#To calculate error
	def error(self, output_var, category):
		err = ((abs(output_var - category)) ** 2)/2
		return err

	#To calculate the dtheta
	# def dtheta(self, output_var, y_target, x_theta):
	# 	dtheta_res = (output_var - y_target)*(1 - output_var) * output_var * x_theta
	# 	return dtheta_res

	def thau_out(self, output_var, y_target):
		thau_o = (output_var - y_target)* output_var * (1 - output_var)
		return thau_o

	def thau_hid(self, thau_1, thau_2, v_theta_1, v_theta_2, output_hid):

		thau_h = ((thau_1 * v_theta_1) + (thau_2 * v_theta_2)) * output_hid * (1 - output_hid)

		return thau_h


# **********************************************************************************************************

thetas_1 = rd.random()
thetas_2 = rd.random()
thetas_3 = rd.random()
thetas_4 = rd.random()

thetas_5 = rd.random()
thetas_6 = rd.random()
thetas_7 = rd.random()
thetas_8 = rd.random()

thetas_9 = rd.random()
thetas_10 = rd.random()
thetas_11 = rd.random()
thetas_12 = rd.random()

thetas_13 = rd.random()
thetas_14 = rd.random()
thetas_15 = rd.random()
thetas_16 = rd.random()

bias_hid_1 = rd.random()
bias_hid_2 = rd.random()
bias_hid_3 = rd.random()
bias_hid_4 = rd.random()

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


v_thetas_1 = rd.random()
v_thetas_2 = rd.random()
v_thetas_3 = rd.random()
v_thetas_4 = rd.random()

v_thetas_5 = rd.random()
v_thetas_6 = rd.random()
v_thetas_7 = rd.random()
v_thetas_8 = rd.random()

bias_v_1 = rd.random()
bias_v_2 = rd.random()


data = IrisData("D:/KULIAH/PELAJARAN_SEMESTER_6/Machine Learning/TUGAS_4/iris.csv", 0)
calc = Calculation(data)

epoch_graph = []

error_graph = []
accuracy_graph =[]

error_v_graph = []
accuracy_v_graph = []

for epoch in range(200):

	error = 0
	accuracy = 0
	
	error_v = 0
	accuracy_v = 0
	# print("\nTHIS IS FOR EPOCH %s\n" % (epoch+1))

	for row in range(120):

		d_thetas_1 = 0
		d_thetas_2 = 0
		d_thetas_3 = 0
		d_thetas_4 = 0

		d_thetas_5 = 0
		d_thetas_6 = 0
		d_thetas_7 = 0
		d_thetas_8 = 0

		d_thetas_9 = 0
		d_thetas_10 = 0
		d_thetas_11 = 0
		d_thetas_12 = 0

		d_thetas_13 = 0
		d_thetas_14 = 0
		d_thetas_15 = 0
		d_thetas_16 = 0

		d_bias_hid_1 = 0
		d_bias_hid_2 = 0
		d_bias_hid_3 = 0
		d_bias_hid_4 = 0
		
		d_v_thetas_1 = 0
		d_v_thetas_2 = 0
		d_v_thetas_3 = 0
		d_v_thetas_4 = 0

		d_v_thetas_5 = 0
		d_v_thetas_6 = 0
		d_v_thetas_7 = 0
		d_v_thetas_8 = 0

		d_v_bias_1 = 0
		d_v_bias_2 = 0

		SP = calc.getSepalPetal(1, row)

		x1 = SP[0]
		x2 = SP[1]
		x3 = SP[2]
		x4 = SP[3]

		category_1 = SP[4]
		category_2 = SP[5]


		#FIRST HITUNG TARGET!!!
		# ADA 4 BRO

		target_1 = (x1 * thetas_1) + (x2 * thetas_2) + (x3 * thetas_3) + (x4 * thetas_4) + bias_hid_1
		target_2 = (x1 * thetas_5) + (x2 * thetas_6) + (x3 * thetas_7) + (x4 * thetas_8) + bias_hid_2
		target_3 = (x1 * thetas_9) + (x2 * thetas_10) + (x3 * thetas_11) + (x4 * thetas_12) + bias_hid_3
		target_4 = (x1 * thetas_13) + (x2 * thetas_14) + (x3 * thetas_15) + (x4 * thetas_16) + bias_hid_4

		#NAH UDAH KAN
		#SEKARANG ITUNG OUTPUT DARI TARGET CUYYY
		#ada 4 juga lah

		output_hid_1 = 1/(1 + mt.exp(-target_1))
		output_hid_2 = 1/(1 + mt.exp(-target_2))
		output_hid_3 = 1/(1 + mt.exp(-target_3))
		output_hid_4 = 1/(1 + mt.exp(-target_4))

		#ITU DAPET OUTPUT YANG HIDDEN

		#TERUS COBA ITUNG TARGET LAGI YANG TERAKHER
		#cuma ada 2 BRO

		target_final_1 = (output_hid_1 * v_thetas_1) + (output_hid_2 * v_thetas_3) + (output_hid_3 * v_thetas_5) + (output_hid_4 * v_thetas_7) + bias_v_1
		target_final_2 = (output_hid_1 * v_thetas_2) + (output_hid_2 * v_thetas_4) + (output_hid_3 * v_thetas_6) + (output_hid_4 * v_thetas_8) + bias_v_2

		output_final_1 = 1/(1 + np.exp(-target_final_1))
		output_final_2 = 1/(1 + np.exp(-target_final_2))

		#MANTAPPP DIKIT LAGI
		#SKARANG MAU ITUNG ERROR DULU
		error_final_1 = (output_final_1 - category_1)**2
		error_final_2 = (output_final_2 - category_2)**2

		#INI BUAT DICOCOKIN NTAR JADI ACCURACYNYA

		if(output_final_1 < 0.5):
			predict_1 = 0
		else:
			predict_1 = 1

		if(output_final_2 < 0.5):
			predict_2 = 0
		else:
			predict_2 = 1

		#TAPI NTAR AJA
		#MAU BACK PROPAGATION DULU

		thau_out_1 = (output_final_1 - category_1) * (output_final_1) * (1 - output_final_1)
		thau_out_2 = (output_final_2 - category_2) * (output_final_2) * (1 - output_final_2)

		#HITUNG DVTHETA CUY
		d_v_thetas_1 = output_hid_1 * thau_out_1
		d_v_thetas_2 = output_hid_1 * thau_out_2

		d_v_thetas_3 = output_hid_2 * thau_out_1
		d_v_thetas_4 = output_hid_2 * thau_out_2

		d_v_thetas_5 = output_hid_3 * thau_out_1
		d_v_thetas_6 = output_hid_3 * thau_out_2

		d_v_thetas_7 = output_hid_4 * thau_out_1
		d_v_thetas_8 = output_hid_4 * thau_out_2

		d_v_bias_1 = thau_out_1
		d_v_bias_2 = thau_out_2		

		#COBA HITUNG THAU HIDDEN NYA COYYYY
		#COBA COBA
		thau_hidden_1 = ((thau_out_1 * v_thetas_1) + (thau_out_2 * v_thetas_2)) * output_hid_1 * (1 - output_hid_1)
		thau_hidden_2 = ((thau_out_1 * v_thetas_3) + (thau_out_2 * v_thetas_4)) * output_hid_2 * (1 - output_hid_2)
		thau_hidden_3 = ((thau_out_1 * v_thetas_5) + (thau_out_2 * v_thetas_6)) * output_hid_3 * (1 - output_hid_3)
		thau_hidden_4 = ((thau_out_1 * v_thetas_7) + (thau_out_2 * v_thetas_8)) * output_hid_4 * (1 - output_hid_4)

		#MAMPUS SKARANG YANG d_theta yang ada 16!!! BANYAK BANGET BOS
		d_thetas_1 = x1 * thau_hidden_1
		d_thetas_2 = x1 * thau_hidden_2
		d_thetas_3 = x1 * thau_hidden_3
		d_thetas_4 = x1 * thau_hidden_4

		d_thetas_5 = x2 * thau_hidden_1
		d_thetas_6 = x2 * thau_hidden_2
		d_thetas_7 = x2 * thau_hidden_3
		d_thetas_8 = x2 * thau_hidden_4

		d_thetas_9 = x3 * thau_hidden_1
		d_thetas_10 = x3 * thau_hidden_2
		d_thetas_11 = x3 * thau_hidden_3
		d_thetas_12 = x3 * thau_hidden_4

		d_thetas_13 = x4 * thau_hidden_1
		d_thetas_14 = x4 * thau_hidden_2
		d_thetas_15 = x4 * thau_hidden_3
		d_thetas_16 = x4 * thau_hidden_4

		d_bias_hid_1 = thau_hidden_1
		d_bias_hid_2 = thau_hidden_2
		d_bias_hid_3 = thau_hidden_3
		d_bias_hid_4 = thau_hidden_4

		#AKHIRNYA!! SMUEA NYA UDA DIHITUNG, SKARANG TINGGAL IMPROVE THETA, BIAS_H, VTHETHA, BIAS V

		thetas_1 = thetas_1 - (0.1 * d_thetas_1)
		thetas_2 = thetas_2 - (0.1 * d_thetas_2)
		thetas_3 = thetas_3 - (0.1 * d_thetas_3)
		thetas_4 = thetas_4 - (0.1 * d_thetas_4)

		thetas_5 = thetas_5 - (0.1 * d_thetas_5)
		thetas_6 = thetas_6 - (0.1 * d_thetas_6)
		thetas_7 = thetas_7 - (0.1 * d_thetas_7)
		thetas_8 = thetas_8 - (0.1 * d_thetas_8)

		thetas_9 = thetas_9 - (0.1 * d_thetas_9)
		thetas_10 = thetas_10 - (0.1 * d_thetas_10)
		thetas_11 = thetas_11 - (0.1 * d_thetas_11)
		thetas_12 = thetas_12 - (0.1 * d_thetas_12)

		thetas_13 = thetas_13 - (0.1 * d_thetas_13)
		thetas_14 = thetas_14 - (0.1 * d_thetas_14)
		thetas_15 = thetas_15 - (0.1 * d_thetas_15)
		thetas_16 = thetas_16 - (0.1 * d_thetas_16)

		bias_hid_1 = thetas_1 - (0.1 * d_bias_hid_1)
		bias_hid_2 = thetas_2 - (0.1 * d_bias_hid_2)
		bias_hid_3 = thetas_3 - (0.1 * d_bias_hid_3)
		bias_hid_4 = thetas_4 - (0.1 * d_bias_hid_4)

		#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


		v_thetas_1 = v_thetas_1 - (0.1 * d_v_thetas_1)
		v_thetas_2 =  v_thetas_2 - (0.1 * d_v_thetas_2)
		v_thetas_3 =  v_thetas_3 - (0.1 * d_v_thetas_3)
		v_thetas_4 =  v_thetas_4 - (0.1 * d_v_thetas_4)

		v_thetas_5 =  v_thetas_5 - (0.1 * d_v_thetas_5)
		v_thetas_6 =  v_thetas_6 - (0.1 * d_v_thetas_6)
		v_thetas_7 =  v_thetas_7 - (0.1 * d_v_thetas_7)
		v_thetas_8 =  v_thetas_8 - (0.1 * d_v_thetas_8)

		bias_v_1 = bias_v_1 - (0.1 * d_v_bias_1)
		bias_v_2 = bias_v_2 - (0.1 * d_v_bias_2)

		error += error_final_1 + error_final_2
		if(predict_1 == category_1 and predict_2 == category_2):
			accuracy += 1

	error = error/120
	accuracy = accuracy/120

	error_graph.append(error)
	accuracy_graph.append(accuracy)
	epoch_graph.append(epoch+1)

	print("%s. Error: %s \t Accuracy: %s\n" % ((epoch+1), error, accuracy))

################################################################
	#SEMUA UDAH SKRANG KITA VALIDASI MAS MAS BRO BRO

	for v_row in range(30):
		SP = calc.getSepalPetal(0, v_row)

		x1_v = SP[0]
		x2_v = SP[1]
		x3_v = SP[2]
		x4_v = SP[3]

		category_1_v = SP[4]
		category_2_v = SP[5]

		#FIRST HITUNG TARGET!!!
		# ADA 4 BRO

		target_1_v = (x1_v * thetas_1) + (x2_v * thetas_2) + (x3_v * thetas_3) + (x4_v * thetas_4) + bias_hid_1
		target_2_v = (x1_v * thetas_5) + (x2_v * thetas_6) + (x3_v * thetas_7) + (x4_v * thetas_8) + bias_hid_2
		target_3_v = (x1_v * thetas_9) + (x2_v * thetas_10) + (x3_v * thetas_11) + (x4_v * thetas_12) + bias_hid_3
		target_4_v = (x1_v * thetas_13) + (x2_v * thetas_14) + (x3_v * thetas_15) + (x4_v * thetas_16) + bias_hid_4

		#NAH UDAH KAN
		#SEKARANG ITUNG OUTPUT DARI TARGET CUYYY
		#ada 4 juga lah

		output_hid_1_v = 1/(1 + mt.exp(-target_1_v))
		output_hid_2_v = 1/(1 + mt.exp(-target_2_v))
		output_hid_3_v = 1/(1 + mt.exp(-target_3_v))
		output_hid_4_v = 1/(1 + mt.exp(-target_4_v))

		#ITU DAPET OUTPUT YANG HIDDEN

		#TERUS COBA ITUNG TARGET LAGI YANG TERAKHER
		#cuma ada 2 BRO

		target_final_1_v = (output_hid_1_v * v_thetas_1) + (output_hid_2_v * v_thetas_3) + (output_hid_3_v * v_thetas_5) + (output_hid_4_v * v_thetas_7) + bias_v_1
		target_final_2_v = (output_hid_1_v * v_thetas_2) + (output_hid_2_v * v_thetas_4) + (output_hid_3_v * v_thetas_6) + (output_hid_4_v * v_thetas_8) + bias_v_2

		output_final_1_v = 1/(1 + np.exp(-target_final_1_v))
		output_final_2_v = 1/(1 + np.exp(-target_final_2_v))

		#MANTAPPP DIKIT LAGI
		#SKARANG MAU ITUNG ERROR DULU
		error_final_1_v = (output_final_1_v - category_1_v)**2
		error_final_2_v = (output_final_2_v - category_2_v)**2

		#INI BUAT DICOCOKIN NTAR JADI ACCURACYNYA

		if(output_final_1_v < 0.5):
			predict_1_v = 0
		else:
			predict_1_v = 1

		if(output_final_2_v < 0.5):
			predict_2_v = 0
		else:
			predict_2_v = 1

		error_v += error_final_1_v + error_final_2_v
		if(predict_1_v == category_1_v and predict_2_v == category_2_v):
			accuracy_v += 1

	error_v = error_v/30
	accuracy_v = accuracy_v/30

	error_v_graph.append(error_v)
	accuracy_v_graph.append(accuracy_v)

	print("    Err_V: %s \t Acc_V: %s\n" % (error_v, accuracy_v))


plt.figure(1)
plt.plot(epoch_graph, error_graph, label = 'Training Error average')
plt.plot(epoch_graph, error_v_graph, label = 'Validation Error average')
plt.title('Average Error Graph Block')
plt.legend(loc = "best")

plt.figure(2)
plt.plot(epoch_graph, accuracy_graph, label = 'Training Accuracy average')
plt.plot(epoch_graph, accuracy_v_graph, label = 'Validation Accuracy average')
plt.title('Average Accuracy Graph Block')
plt.legend(loc = "best")

plt.show()

# vtheta[0] = v11	vtheta[4] = v12
# vtheta[1] = v21 	vtheta[5] = v22
# vtheta[2] = v31 	...
# vtheta[3] = v41

