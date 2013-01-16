/**
 * Created with IntelliJ IDEA.
 * User: wind
 * Date: 13-1-15
 * To change this template use File | Settings | File Templates.
 */

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.TextDirectoryLoader;
import weka.core.stemmers.NullStemmer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.*;

public class PropertyClassifier {

    public static void main(String args[]) throws IOException {
        System.out.println("hello wind!\n");

        //PythonInterpreter interpreter = new PythonInterpreter();

        //interpreter.exec("print(\"called from java!\\n\")");

        try {
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
        }
        // Convert directory-based text classes to Weka's raw-input type(unfiltered)
        String dirPath = "D:\\Documents\\Projects\\Description Classifier\\data\\property";
        String targetedFilePath = "D:\\Documents\\Projects\\Description Classifier\\data\\dataRaw.arff";
        textDirectoryToRawFile(dirPath, targetedFilePath);

        // Raw File filter: String to Word Vectors

        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(targetedFilePath));
        Instances dataRaw = loader.getDataSet();


        StringToWordVector filter = new StringToWordVector();
        filter.setStemmer(new NullStemmer());
        try {
            filter.setInputFormat(dataRaw);
        } catch (Exception e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }
        try {
            Instances dataFilter = Filter.useFilter(dataRaw, filter);
        } catch (Exception e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }
//        Classifier classifier = (Classifier) new NaiveBayesMultinomialUpdateable();
    }

    public static void textDirectoryToRawFile(String dirPath, String targetedFilePath) {
        TextDirectoryLoader textDirectoryLoader = new TextDirectoryLoader();
        try {
            textDirectoryLoader.setDirectory(new File(dirPath));
            Instances dataRaw = textDirectoryLoader.getDataSet();
            {
                FileWriter arffFileWriter = new FileWriter(targetedFilePath);
                BufferedWriter bw = new BufferedWriter(arffFileWriter);

                bw.write(dataRaw.toString());
                bw.close();
                arffFileWriter.close();
            }
        } catch (IOException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }
    }
}
