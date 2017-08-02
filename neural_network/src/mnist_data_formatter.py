import numpy as np
import mnist_loader

training_data, validation_data, test_data= \
mnist_loader.load_data_wrapper()

n_train=len(training_data)
n_validation=len(validation_data)
n_test=len(test_data)

#format training data and save into train_input/output.npy
train_input=training_data[0][0]
train_output=training_data[0][1]
for i in xrange(1,n_train):
   	train_input=np.concatenate((train_input,training_data[i][0]),axis=1)
	train_output=np.concatenate((train_output,training_data[i][1]),axis=1)
	if i%99==0: print "Formatting training data ......",100.0*float(i)/float(n_train),"%"

train_input=train_input.T
train_output=train_output.T

np.save("train_input.npy",train_input)
np.save("train_output.npy",train_output)

print "FORMATTING TRAINING DATA SUCCESSFUL"

#format validation data and save into validation_input/output.npy
validation_input=validation_data[0][0]
validation_output=np.array([])
validation_output=np.append(validation_output,validation_data[0][1])
for i in xrange(1,n_validation):
        validation_input=np.concatenate((validation_input,validation_data[i][0]),axis=1)
        validation_output=np.append(validation_output,validation_data[i][1])
        if i%99==0: print "Formatting validation data ......",100.0*float(i)/float(n_validation),"%"

validation_input=validation_input.T
validation_output=validation_output

np.save("validation_input.npy",validation_input)
np.save("validation_output.npy",validation_output)

print "FORMATTING VALIDATION DATA SUCCESSFUL"

#format test data and save into validation_input/output.npy
test_input=test_data[0][0]
test_output=np.array([])
test_output=np.append(test_output,test_data[0][1])
for i in xrange(1,n_test):
        test_input=np.concatenate((test_input,test_data[i][0]),axis=1)
        test_output=np.append(test_output,test_data[i][1])
        if i%99==0: print "Formatting test data ......",100.0*float(i)/float(n_test),"%"

test_input=test_input.T
test_output=test_output

np.save("test_input.npy",test_input)
np.save("test_output.npy",test_output)

print "FORMATTING TEST DATA SUCCESSFUL"
print "---FORMATTING DONE---"
