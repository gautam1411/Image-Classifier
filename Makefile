JC = javac
.SUFFIXES: .java .class
.java.class:
	$(JC) $*.java

CLASSES = \
	ConvNet.java \
	Convolution.java \
	Dataset.java \
	FeatureMap.java \
	FlatLayer.java \
	Instance.java \
	Lab3DNN.java \
	OutputLayer.java \
	PoolMap.java \
	Pooling.java

default: classes

classes: $(CLASSES:.java=.class)

clean:
	$(RM) *.class *~
