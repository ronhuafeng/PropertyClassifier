/**
 * Created with IntelliJ IDEA.
 * User: wind
 * Date: 13-1-15
 * To change this template use File | Settings | File Templates.
 */

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.TextDirectoryLoader;
import weka.core.stemmers.NullStemmer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Enumeration;
import java.util.Random;

public class PropertyClassifier {

	public static void main(String args[]) throws IOException {
		classifyProcessLineDemo();


	}

	private static void classifyProcessLineDemo() throws IOException {
		// Convert directory-based text classes to Weka's raw-input type(unfiltered)
		String dirPath = "D:\\Documents\\Projects\\Description Classifier\\data\\property";
		String targetedFilePath = "D:\\Documents\\Projects\\Description Classifier\\data\\dataRaw.arff";
		textDirectoryToRawFile(dirPath, targetedFilePath);

		// Raw File filter: String to Word Vectors

		//  * Read the file got in last step into object.
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(targetedFilePath));
		Instances dataRaw = loader.getDataSet();


		StringToWordVector filter = new StringToWordVector();
		String filteredFilePath = "D:\\Documents\\Projects\\Description Classifier\\data\\dataFiltered.arff";
		filter.setStemmer(new NullStemmer());
		try {
			filter.setInputFormat(dataRaw);
			Instances dataFiltered = Filter.useFilter(dataRaw, filter);
			writeStringToFile(filteredFilePath, dataFiltered.toString());

			dataFiltered.setClassIndex(0);
		/* ARFF文件不保存类的属性信息（XRFF保存）
        * 可以通过setClassIndex(int)方法来设定类别
        * 0表示使用数据集的第一属性作为类别属性
        * data.numAttributes() - 1 表示使用数据集的最后一个属性作为类别属性
        *      示例代码       *
        *      if (data.classIndex() == -1)
        *           data.setClassIndex(0);
        *      if (data.classIndex() == -1)
        *           data.setClassIndex(data.numAttributes() - 1);
        * */

			// * test the classifier module
			// * * test: saved filtered file can be restored
			loader.setFile(new File(filteredFilePath));
			dataFiltered = loader.getDataSet();
			dataFiltered.setClassIndex(0);

			// * * begin to classify
			NaiveBayes classifier = new NaiveBayes();
			classifier.buildClassifier(dataFiltered);

			Evaluation evaluation = new Evaluation(dataFiltered);
			evaluation.crossValidateModel(classifier, dataFiltered, 10, new Random(1));

			System.out.println(evaluation.toClassDetailsString());
			System.out.println(evaluation.toSummaryString());
			System.out.println(evaluation.toMatrixString());


			// * Save model to disk：通过序列化的方法
			String modelFilePath = "D:\\Documents\\Projects\\Description Classifier\\data\\naive_bayes.model";
			SerializationHelper.write(modelFilePath, classifier);


			// * read model from disk
			classifier = (NaiveBayes) SerializationHelper.read(modelFilePath);

			// * test for classifier precision
			for (int i = 0; i < dataFiltered.numInstances(); i++) {
				double prediction = classifier.classifyInstance(dataFiltered.instance(i));
				String category = dataFiltered.classAttribute().value((int) prediction);

				String actualCategory = dataFiltered.classAttribute().value((int) dataFiltered.instance(i).classValue());
				System.out.println(actualCategory + " belongs to: " + category);
			}

			// * extract words in word-vector used by classifier
			System.out.println(dataFiltered.attribute(55));
			dataFiltered.enumerateAttributes();
			Enumeration<Attribute> attributeEnumeration = dataFiltered.enumerateAttributes();

			while (attributeEnumeration.hasMoreElements()) {
				Attribute attribute = attributeEnumeration.nextElement();
				// * After a few tests, I confirm that the name field has the right word value.
				System.out.println(attribute.name());
			}

			// *   把一个句子转变成可供训练集得到的分类器使用的一个arff文件
			String testsetFilePath = "D:\\Documents\\Projects\\Description Classifier\\data\\property_extended.arff";
			Instances convertedInstances = getConvertedInstances(filteredFilePath, testsetFilePath, loader);

			int rightPrediction = 0;
			for (int i = 0; i < convertedInstances.numInstances(); i++) {
				String predictedCategory = dataFiltered.classAttribute().value((int) classifier.classifyInstance(convertedInstances.instance(i)));

				if (convertedInstances.instance(i).attribute(0).value(0).equals(predictedCategory)) {
					rightPrediction++;
				}
				else {
					System.out.println(convertedInstances.instance(i).attribute(0).value(0) + " : " + predictedCategory);
				}
			}

			System.out.println(rightPrediction + " / " + convertedInstances.numInstances());

		} catch (Exception e) {
			e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
		}
	}

	public static Instances getConvertedInstances(String filteredFilePath, String testsetFilePath, ArffLoader loader) throws Exception {
		// * Read testset file to object
		loader.setFile(new File(testsetFilePath));
		Instances testsetInstances = loader.getDataSet();
		testsetInstances.setClassIndex(0);

		// * Consturct the converted instances object
		loader.setFile(new File(filteredFilePath));
		Instances dataFilteredHeader = loader.getStructure();
		dataFilteredHeader.setClassIndex(0);
		Instances convertedInstances = new Instances(dataFilteredHeader);
		convertedInstances.setClassIndex(0);

		for (int i = 0; i < testsetInstances.numInstances(); i++) {
			Instance originInstance = testsetInstances.instance(i);

			// * instance for class prediction
			Instance convertedInstance = new Instance(dataFilteredHeader.numAttributes());
			convertedInstance.setDataset(dataFilteredHeader);

			// * Loop attributes in the original data set, add the corresponding (attribute, value) to the new instance
			for (int j = 0; j < dataFilteredHeader.numAttributes(); j++) {
				Attribute attributeInFilteredSet = dataFilteredHeader.attribute(j);

//					// * attribute exists in the original instance?
//					System.out.println(attribute.toString() + "    " + attribute.index());

				// * Find the attribute in the testset which has the same name as in the filtered set
				Attribute attributeInTestSet = dataFilteredHeader.attribute(attributeInFilteredSet.name());
				if (attributeInTestSet != null) {
					if (attributeInTestSet.toString().equals(attributeInFilteredSet.toString()) == false) {
						// * 其实类型不同是可以进行转换的，不一定要抛出异常；这里是偷懒了
						throw new Exception(
								"属性不能转换，类型不匹配:\n"
										+ attributeInTestSet.toString() + "\n"
										+ attributeInFilteredSet.toString()
						);
					}

					// * 不知道是否这样就可以了，上面的判断应该可以保证类型是匹配的了
					convertedInstance.setValue(attributeInFilteredSet, originInstance.value(attributeInTestSet));
				}
			}

			convertedInstances.add(convertedInstance);
		}
		return convertedInstances;
	}

	public static void textDirectoryToRawFile(String dirPath, String targetedFilePath) {
		TextDirectoryLoader textDirectoryLoader = new TextDirectoryLoader();
		try {
			textDirectoryLoader.setDirectory(new File(dirPath));
			Instances dataRaw = textDirectoryLoader.getDataSet();
			writeStringToFile(targetedFilePath, dataRaw.toString());

		} catch (IOException e) {
			e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
		}
	}

	private static void writeStringToFile(String targetedFilePath, String text) throws IOException {
		FileWriter arffFileWriter = new FileWriter(targetedFilePath);
		BufferedWriter bw = new BufferedWriter(arffFileWriter);

		bw.write(text);
		bw.close();
		arffFileWriter.close();
	}
}
