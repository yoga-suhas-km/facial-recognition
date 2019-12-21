import sys
import our_model
import model_data_set
import model_global_variable

def main():
    sys.stdout.write('Welcome to CIFAR-10 Hello world of CONVNET!\n')
    sys.stdout.write('Here we try to build a more advanced model\n\n')
    sys.stdout.flush()
    X_train, y_train, X_test, y_test = model_data_set.get_preprocessed_dataset()
    model = our_model.generate_model()
    model = our_model.train(model, X_train, y_train, X_test, y_test)
    # this works for raw data.. 
    #todo: find the way to find performance metrics for image related issues..
    #our_model.print_performance_metrics(model, X_test, y_test, model_global_variable.batch_size)

if __name__ == "__main__":
    # execute only if run as a script
    main()