use debot_db::ModelParams;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;

pub struct RandomForest {
    model: RandomForestClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>,
}

impl RandomForest {
    pub async fn new(key: &str, model_params: &ModelParams) -> Self {
        let serializable_model = model_params
            .load_model(key)
            .await
            .expect("Failed to load model");
        let model: RandomForestClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>> =
            bincode::deserialize(&serializable_model.model).unwrap();

        Self { model }
    }

    pub fn predict(&self, x: DenseMatrix<f64>) -> bool {
        let prediction = self.model.predict(&x).unwrap();
        log::info!("Prediction: {:?}", prediction);
        if prediction.len() == 1 {
            prediction[0] == 1
        } else {
            false
        }
    }
}
