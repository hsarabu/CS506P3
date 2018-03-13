import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Fill in the implementation details of the class DecisionTree using this file. Any methods or
 * secondary classes that you want are fine but we will only interact with those methods in the
 * DecisionTree framework.
 * <p>
 * You must add code for the 1 member and 4 methods specified below.
 * <p>
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
    private DecTreeNode root;
    //ordered list of class labels
    private List<String> labels;
    //ordered list of attributes
    private List<String> attributes;
    //map to ordered discrete values taken by attributes
    private Map<String, List<String>> attributeValues;
    //map for getting the index
    private HashMap<String, Integer> label_inv;
    private HashMap<String, Integer> attr_inv;

    /**
     * Answers static questions about decision trees.
     */
    DecisionTreeImpl() {
        // no code necessary this is void purposefully
    }

    /**
     * Build a decision tree given only a training set.
     *
     * @param train: the training set
     */
    DecisionTreeImpl(DataSet train) {

        this.labels = train.labels;
        this.attributes = train.attributes;
        this.attributeValues = train.attributeValues;
        // TODO: Homework requirement, learn the decision tree here
        // Get the list of instances via train.instances
        // You should write a recursive helper function to build the tree
        //
        // this.labels contains the possible labels for an instance
        // this.attributes contains the whole set of attribute names
        // train.instances contains the list of instances

        root = decisionTreeImplHelper(train.instances,attributes, majorityLabel(train.instances), null, null);
    }

    DecTreeNode decisionTreeImplHelper(List<Instance> instances, List<String> attributes, String defaultLabel, String attribute, String parentAttributeVal){

        if(instances.isEmpty()){
            //TODO: Think I have to change those nulls, will leave them for now
            return new DecTreeNode(defaultLabel, attribute, parentAttributeVal, true);
        }
        if(sameLabel(instances)){
            //TODO: Probably have to change those nulls
            return new DecTreeNode(majorityLabel(instances), attribute, parentAttributeVal, true);
        }
        if(attributes.isEmpty()){
            return new DecTreeNode(majorityLabel(instances), attribute, parentAttributeVal, true);
        }
        //calculate best info gain from the list of attributes
        String bestAttr = attributes.get(0);
        double maxInfoGain = InfoGain(instances, bestAttr);
        for(String attr: attributes){
            double infoGain = InfoGain(instances, attr);
            if(infoGain > maxInfoGain){
                maxInfoGain = infoGain;
                bestAttr = attr;
            }
        }
        DecTreeNode curr = new DecTreeNode(majorityLabel(instances), bestAttr, parentAttributeVal, false);
        for(String attr: attributeValues.get(bestAttr)){
            //v-ex = subset of examples with q == v
            //subtree = buildtree(v-ex, attributes - {q}, majority-class(examples))
            //add arc from tree to subtree
            List<Instance> vEX = instances.stream().filter(q -> q.attributes.contains(attr)).collect(Collectors.toList());
            ArrayList<String> newAttributes = new ArrayList<>(attributes);
            newAttributes.remove(bestAttr);
            DecTreeNode subtree;
            if(vEX.isEmpty()){
                subtree = decisionTreeImplHelper(vEX, newAttributes, defaultLabel,bestAttr, attr);
            }
            else {
                String majorityLabel = majorityLabel(vEX);
                subtree = decisionTreeImplHelper(vEX, newAttributes, majorityLabel,bestAttr, attr);
            }
            curr.children.add(subtree);
        }

        return curr;
    }

    boolean sameLabel(List<Instance> instances) {
        // Suggested helper function
        // returns if all the instances have the same label
        // labels are in instances.get(i).label
        Instance first = instances.get(0);
        for(Instance curr: instances){
            if(first.label.equals(curr.label)) continue;
            else return false;
        }
        return true;
    }

    String majorityLabel(List<Instance> instances) {
        // Suggested helper function
        // returns the majority label of a list of examples

        //create a hashmap for the labels
        HashMap<String, Integer> labels = new HashMap<>();

        //creating shadow arraylist for tie condition
        ArrayList<String> labelOrder = new ArrayList<>();
        for(Instance curr: instances){
            if(labels.containsKey(curr.label)) labels.put(curr.label, labels.get(curr.label) + 1);
            else {
                labels.put(curr.label, 1);
                labelOrder.add(curr.label);
            }
        }

        //GREEDY BOI

        int maxValue = 0;
        String maxLabel = "";
        for(String label: labelOrder){
            int tempValue = labels.get(label);
            if(tempValue > maxValue) {
                maxValue = tempValue;
                maxLabel = label;
            }
        }

        return maxLabel;
    }

    double entropy(List<Instance> instances) {
        // Suggested helper function
        // returns the Entropy of a list of examples


        double totalSize = instances.size();


        //use part of majority label code to get counts of all the labels
        //create a hashmap for the labels
        HashMap<String, Integer> labels = new HashMap<>();

        for(Instance curr: instances){
            if(labels.containsKey(curr.label)) labels.put(curr.label, labels.get(curr.label) + 1);
            else {
                labels.put(curr.label, 1);
            }
        }
        //don't really care about the label, just the values
        double entropy = 0.0;
        for(double value: labels.values()){
            double probability = value/totalSize;
            if(probability != 0) {
                double log = Math.log(probability) / Math.log(2); // log10 / log 2 makes log base 2
                //TODO: Check if entropy forumal is negative
                entropy = entropy - (log * probability);
            }
        }

        return entropy;
    }

    double conditionalEntropy(List<Instance> instances, String attr) {
        // Suggested helper function
        // returns the conditional entropy of a list of examples, given the attribute attr

        //copy some of the maxLabelCount function code to get counts of labels and list of labels
        //create a hashmap for the labels
        double entropy = 0.0;
        int yes = 0, no = 0;
        int count[] = new int[attributeValues.get(attr).size()];
        int total[] = new int[attributeValues.get(attr).size()];
        for(Instance curr: instances){
            String value = curr.attributes.get(getAttributeIndex(attr));
            int index = getAttributeValueIndex(attr, value);
            if(curr.label.equals(this.labels.get(0))) count[index]++;
            total[index]++;
        }
        //yes and no are arbitrary
        //variables to use to calculate conditional entropy
        double weight;
        double probYes = 0.0;
        double probNo = 0.0;
        for(int i = 0; i < attributeValues.get(attr).size(); i++){
            weight = total[i] * 1.0/instances.size();

            if(total[i] != 0){
                probYes = 1.0 * count[i]/total[i];
                probNo = 1 - probYes;
            }
            double logYes =0.0, logNo=0.0;
            if(probYes != 0) logYes = Math.log(probYes) * 1.0 / Math.log(2);

            if(probNo != 0) logNo = Math.log(probNo) * 1.0 / Math.log(2);
            entropy = entropy + weight * -1.0 * (probYes*logYes + probNo*logNo);
        }

        /*
        double overallProbablity = (yes + no) *1.0 /instances.size();

        //we use this in to diving yes.size and no.size by
        int totalInInstance = yes + no;
        double logYes =0.0, logNo=0.0;

        double yesProb = yes * 1.0 / totalInInstance;
        if(yesProb != 0) logYes = Math.log(yesProb) * 1.0 / Math.log(2);

        double noProb = no * 1.0 / totalInInstance;
        if(noProb != 0) logNo = Math.log(noProb) * 1.0 / Math.log(2);

        entropy = overallProbablity * -1.0 * (yesProb*logYes + noProb*logNo);

*/

        return entropy;
    }

    double InfoGain(List<Instance> instances, String attr) {
        // Suggested helper function
        // returns the info gain of a list of examples, given the attribute attr
        return entropy(instances) - conditionalEntropy(instances, attr);
    }

    @Override
    public String classify(Instance instance) {
        // TODO: Homework requirement
        // The tree is already built, when this function is called
        // this.root will contain the learnt decision tree.
        // write a recusive helper function, to return the predicted label of instance

        return classifyHelper(instance, this.root);
    }
    public String classifyHelper(Instance instance, DecTreeNode curr){
        //read if current node is terminal
        if(curr.terminal) return curr.label;
        //otherwise go down the tree following the right path
        for(DecTreeNode child: curr.children){
            if(instance.attributes.contains(child.attribute)) return classifyHelper(instance, child);
        }
        return curr.label;
    }

    @Override
    public void rootInfoGain(DataSet train) {
        this.labels = train.labels;
        this.attributes = train.attributes;
        this.attributeValues = train.attributeValues;
        // TODO: Homework requirement
        // Print the Info Gain for using each attribute at the root node
        // The decision tree may not exist when this funcion is called.
        // But you just need to calculate the info gain with each attribute,
        // on the entire training set.
        for(String attr: attributes) {
            System.out.format("%.5f\n", InfoGain(train.instances, attr));
        }
    }

    @Override
    public void printAccuracy(DataSet test) {
        // TODO: Homework requirement
        // Print the accuracy on the test set.
        // The tree is already built, when this function is called
        // You need to call function classify, and compare the predicted labels.
        // List of instances: test.instances
        // getting the real label: test.instances.get(i).label
        double right = 0.0;
        double total = 0.0;
        for(Instance instance: test.instances){
            total++;
            String predictedLabel = classify(instance);
            if(predictedLabel.equals(instance.label)) right++;
        }
        System.out.format("%.5f\n", right/total);
        //print();

    }

    @Override
    /**
     * Print the decision tree in the specified format
     * Do not modify
     */
    public void print() {

        printTreeNode(root, null, 0);
    }

    /**
     * Prints the subtree of the node with each line prefixed by 4 * k spaces.
     * Do not modify
     */
    public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < k; i++) {
            sb.append("    ");
        }
        String value;
        if (parent == null) {
            value = "ROOT";
        } else {
            int attributeValueIndex = this.getAttributeValueIndex(parent.attribute, p.parentAttributeValue);
            value = attributeValues.get(parent.attribute).get(attributeValueIndex);
        }
        sb.append(value);
        if (p.terminal) {
            sb.append(" (" + p.label + ")");
            System.out.println(sb.toString());
        } else {
            sb.append(" {" + p.attribute + "?}");
            System.out.println(sb.toString());
            for (DecTreeNode child : p.children) {
                printTreeNode(child, p, k + 1);
            }
        }
    }

    /**
     * Helper function to get the index of the label in labels list
     */
    private int getLabelIndex(String label) {
        if (label_inv == null) {
            this.label_inv = new HashMap<String, Integer>();
            for (int i = 0; i < labels.size(); i++) {
                label_inv.put(labels.get(i), i);
            }
        }
        return label_inv.get(label);
    }

    /**
     * Helper function to get the index of the attribute in attributes list
     */
    private int getAttributeIndex(String attr) {
        if (attr_inv == null) {
            this.attr_inv = new HashMap<String, Integer>();
            for (int i = 0; i < attributes.size(); i++) {
                attr_inv.put(attributes.get(i), i);
            }
        }
        return attr_inv.get(attr);
    }

    /**
     * Helper function to get the index of the attributeValue in the list for the attribute key in the attributeValues map
     */
    private int getAttributeValueIndex(String attr, String value) {
        for (int i = 0; i < attributeValues.get(attr).size(); i++) {
            if (value.equals(attributeValues.get(attr).get(i))) {
                return i;
            }
        }
        return -1;
    }
}
