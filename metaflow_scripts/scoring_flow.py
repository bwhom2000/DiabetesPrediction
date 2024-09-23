from metaflow import FlowSpec, step, Parameter
import data_processing
import pandas as pd

class ScoringFlow(FlowSpec):
    # Parameters
    data_path = Parameter('data_path', help='Path to the new data')
    model_name = Parameter('model_name', help='Name of the model registered in MLflow')


    @step
    def start(self):
        """
        Initial step to load the new data from the specified path.
        """
        # Load new data
        self.data = data_processing.load_data(self.data_path)
        self.next(self.transform_data)


    @step
    def transform_data(self):
        """
        Transform the new data by applying one-hot encoding technique.
        """
        # Perform feature transformations on the new data
        self.processed_data = data_processing.process_data(self.data, cols_to_encode=['gender', 'smoking_history'])
        self.next(self.load_model)


    @step
    def load_model(self):
        """
        Load the registered model from MLflow.
        """
        import mlflow

        mlflow.set_tracking_uri('https://mlflow-service-1028778413110.us-west2.run.app')

        # Load the registered model using its name and version
        self.model = mlflow.pyfunc.load_model(f'models:/{self.model_name}/1')
        self.next(self.make_predictions)


    @step
    def make_predictions(self):
        """
        Make predictions on the transformed data using the loaded model.
        """
        # Make predictions using the loaded model
        self.predictions = self.model.predict(self.processed_data)
        self.next(self.output_predictions)


    @step
    def output_predictions(self):
        """
        Output the predictions to a CSV file.
        """
        # Create a DataFrame for predictions and save it to a CSV file
        predictions_df = pd.DataFrame(self.predictions, columns=['prediction'])
        predictions_df.to_csv('predictions/predictions.csv', index=False)
        print("Predictions saved to predictions/predictions.csv")
        self.next(self.end)


    @step
    def end(self):
        """
        Final step to signal end of flow
        """
        print("Scoring flow complete.")


if __name__ == '__main__':
    ScoringFlow()
