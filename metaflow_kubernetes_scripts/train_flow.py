from metaflow import FlowSpec, step, Parameter, kubernetes, catch, timeout, retry, conda_base
import data_processing
import model_training

@conda_base(libraries={
    'numpy': '1.23.5',
    'scikit-learn': '1.2.2',
    'pandas': '1.5.3',
    'mlflow': '2.15.1'
}, python='3.9.16')
class TrainFlow(FlowSpec):
    # Parameters
    random_seed = Parameter('random_seed', default=39)
    test_size = Parameter('test_size', default=0.2)


    @step
    def start(self):
        """
        Initial step to load the data.
        """
        self.data_path = '../data/raw_data.csv'
        self.target_name = 'diabetes'
        self.cols_to_encode = ['gender', 'smoking_history']

        # Load data from CSV file
        self.data = data_processing.load_data(self.data_path)
        self.next(self.process_data)


    @step
    def process_data(self):
        """
        Process the data by encoding categorical variables and splitting into train/test sets.
        """
        # Process the data by encoding specified columns
        self.processed_data = data_processing.process_data(self.data, self.cols_to_encode)

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = data_processing.split_data(
            self.processed_data, self.target_name, self.test_size, self.random_seed
        )


        self.next(self.train_logistic_regression, self.train_decision_tree, self.train_random_forest)


    @timeout(seconds=3600)
    @kubernetes(cpu=2, memory=4096)
    @retry(times=2)
    @step
    def train_logistic_regression(self):
        """
        Train a logistic regression model and evaluate its performance.
        """
        # Train the logistic regression model
        self.logistic_model = model_training.train_logistic_regression(self.X_train, self.y_train, self.random_seed)

        # Evaluate the model's performance
        evaluation_results = model_training.evaluate_model(self.logistic_model, self.X_test, self.y_test)
        self.logistic_accuracy, self.logistic_precision, self.logistic_recall = evaluation_results


        self.next(self.compare_models)


    @timeout(seconds=3600)
    @kubernetes(cpu=2, memory=4096)
    @retry(times=2)
    @step
    def train_decision_tree(self):
        """
        Train decision tree models with various hyperparameters and evaluate their performance.
        """
        # Get hyperparameter combinations
        dt_params = model_training.get_dt_params()
        self.dt_models = []
        self.dt_accuracies = []
        self.dt_precisions = []
        self.dt_recalls = []

        # Train and evaluate models for each parameter combination
        for param in dt_params:
            model = model_training.train_decision_tree(self.X_train, self.y_train, param)
            accuracy, precision, recall = model_training.evaluate_model(model, self.X_test, self.y_test)

            # Store results
            self.dt_models.append(model)
            self.dt_accuracies.append(accuracy)
            self.dt_precisions.append(precision)
            self.dt_recalls.append(recall)


        self.next(self.compare_models)


    @timeout(seconds=3600)
    @kubernetes(cpu=2, memory=4096)
    @retry(times=2)
    @step
    def train_random_forest(self):
        """
        Train random forest models with various hyperparameters and evaluate their performance.
        """
        # Get hyperparameter combinations
        rf_params = model_training.get_rf_params()
        self.rf_models = []
        self.rf_accuracies = []
        self.rf_precisions = []
        self.rf_recalls = []

        # Train and evaluate models for each parameter combination
        for param in rf_params:
            model = model_training.train_random_forest(self.X_train, self.y_train, param)
            accuracy, precision, recall = model_training.evaluate_model(model, self.X_test, self.y_test)

            # Store results
            self.rf_models.append(model)
            self.rf_accuracies.append(accuracy)
            self.rf_precisions.append(precision)
            self.rf_recalls.append(recall)


        self.next(self.compare_models)

    @catch
    @step
    def compare_models(self, inputs):
        """
        Compare the performance of all trained models and identify the best one.
        """
        # Collect all recalls, models, and their parameters
        self.models = [inputs.train_logistic_regression.logistic_model] + inputs.train_decision_tree.dt_models + inputs.train_random_forest.rf_models
        self.recalls = [inputs.train_logistic_regression.logistic_recall] + inputs.train_decision_tree.dt_recalls + inputs.train_random_forest.rf_recalls
        self.params = [None] + model_training.get_dt_params() + model_training.get_rf_params()

        # Identify the best model based on recall
        best_index = self.recalls.index(max(self.recalls))
        self.best_model = self.models[best_index]
        self.best_recall = self.recalls[best_index]
        self.best_params = self.params[best_index]

        print(f"Best model recall: {self.best_recall}")
        print(f"Best model parameters: {self.best_params}")
        self.next(self.register_model)


    @step
    def register_model(self):
        """
        Register the best model in MLflow.
        """
        import mlflow
        mlflow.set_tracking_uri('https://mlflow-service-1028778413110.us-west2.run.app')
        mlflow.set_experiment('metaflow-experiment-kubernetes')

        # Log the best model and its details to MLflow
        model_training.log_model(self.best_model, self.best_recall, self.best_params)
        self.next(self.end)


    @step
    def end(self):
        """
        Final step to indicate the end of the training process.
        """
        print(f"Training complete. Best model recall: {self.best_recall}")


if __name__ == '__main__':
    TrainFlow()
