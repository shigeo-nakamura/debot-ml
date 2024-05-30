use debot_db::ModelParams;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;

pub struct RandomForest {
    model_0: RandomForestClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>,
    model_1: RandomForestClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>,
}

impl RandomForest {
    pub async fn new(key: &str, model_params: &ModelParams) -> Self {
        let serializable_model_0 = model_params
            .load_model(&format!("{}_0", key))
            .await
            .expect("Failed to load model 0");
        let model_0: RandomForestClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>> =
            bincode::deserialize(&serializable_model_0.model).unwrap();

        let serializable_model_1 = model_params
            .load_model(&format!("{}_1", key))
            .await
            .expect("Failed to load model 1");
        let model_1: RandomForestClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>> =
            bincode::deserialize(&serializable_model_1.model).unwrap();

        Self { model_0, model_1 }
    }

    pub fn predict(&self, x: DenseMatrix<f64>) -> bool {
        let prediction = self.model_0.predict(&x).unwrap();
        log::trace!("First prediction: {:?}", prediction);
        if prediction[0] == 1 {
            let prediction = self.model_1.predict(&x).unwrap();
            log::info!("Second prediction: {:?}", prediction);
            prediction[0] == 1
        } else {
            false
        }
    }
}
