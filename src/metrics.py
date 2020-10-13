def precision(cm_test):
    labels_2 = 0
    sum = 0
    for row in cm_test:
        for i in row:
            sum += i
        print("The precision of class %d is %d/%d: %f"%(labels_2,row[labels_2], sum, (row[labels_2]/sum)))
        sum = 0
        labels_2 = labels_2+1

def recall(cm_test):
    sum = 0
    for label in range(0,len(cm_test)):
        for row in cm_test:
            sum += row[label]
        print("The recall of class %d is %d/%d: %f"%(label,cm_test[label,label], sum, (cm_test[label,label]/sum)))
        sum = 0

def f1_measure(cm_test):
    labels_2 = 0
    sum_recall = 0
    sum_precision = 0
    for row in cm_test:
        for i in row:
            sum_precision += i
        for row_2 in cm_test:
            sum_recall += row_2[labels_2]
        print("The f1-measure of class %d is (2x%fx%f)/(%f+%f): %f"%(labels_2,(row[labels_2]/sum_precision), (cm_test[labels_2,labels_2]/sum_recall), (row[labels_2]/sum_precision), (cm_test[labels_2,labels_2]/sum_recall), 2*(row[labels_2]/sum_precision)*(cm_test[labels_2,labels_2]/sum_recall)/((row[labels_2]/sum_precision)+(cm_test[labels_2,labels_2]/sum_recall))))
        sum_precision = 0
        sum_recall = 0
        labels_2 = labels_2+1