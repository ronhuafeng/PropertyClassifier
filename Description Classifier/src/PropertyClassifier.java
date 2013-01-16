/**
 * Created with IntelliJ IDEA.
 * User: wind
 * Date: 13-1-15
 * To change this template use File | Settings | File Templates.
 */

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
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
        System.out.println("hello wind!\n");

        //PythonInterpreter interpreter = new PythonInterpreter();

        //interpreter.exec("print(\"called from java!\\n\")");

        // Try to tidy out a hand-collected data txt
        /*try {
            String filePath = "D:\\Documents\\Projects\\Description Classifier\\data\\processed_bayes.txt";
            String outputFilePath = "D:\\Documents\\Projects\\Description Classifier\\data\\processed_bayes_output.txt";

            OutputStreamWriter outputStreamWriter = new OutputStreamWriter(new FileOutputStream(outputFilePath));
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filePath));

            String tmpString;
            while ((tmpString = bufferedReader.readLine()) != null) {
                String[] lines = tmpString.split(" +");

                tmpString = "";
                for (String str : lines) {
                    tmpString += str + " ";
                }
                outputStreamWriter.write(tmpString.trim() + "\n");
            }
            bufferedReader.readLine();

            outputStreamWriter.close();
            bufferedReader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }*/
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
            // Why: See the reason below.

            // * test the classifier module
            NaiveBayes classifier = new NaiveBayes();
            classifier.buildClassifier(dataFiltered);

            Evaluation evaluation = new Evaluation(dataFiltered);
            evaluation.crossValidateModel(classifier, dataFiltered, 10, new Random(1));

            System.out.println(evaluation.toClassDetailsString());
            System.out.println(evaluation.toSummaryString());
            System.out.println(evaluation.toMatrixString());


            // * Save model to disk
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

        } catch (Exception e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }

        // ARFF stores no information about class attribute, but XRFF does).
/*        it can be set with the setClassIndex(int) method:
        // uses the first attribute as class attribute
                if (data.classIndex() == -1)
                    data.setClassIndex(0);
                ...
        // uses the last attribute as class attribute
                if (data.classIndex() == -1)
                    data.setClassIndex(data.numAttributes() - 1);*/


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
