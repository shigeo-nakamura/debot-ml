use tangram::Model;
use tangram::PredictOutput;

fn main() {
    // トレーニングされたモデルをファイルからロードします。
    let model: Model = Model::from_path("heart_disease.tangram", None).unwrap();

    // 予測のためのデータを作成します。
    let input = tangram::predict_input! {
        "age": 63.0,
        "gender": "male",
        // 他の特徴量も同様にここに追加してください。
    };

    // 予測を実行します。
    let output = model.predict_one(input, None);

    // 予測結果を表示します。
    match output {
        PredictOutput::Regression(output) => println!("Prediction: {}", output.value),
        PredictOutput::BinaryClassification(output) => {
            println!(
                "Prediction: probability of positive class = {}",
                output.probability
            )
        }
        PredictOutput::MulticlassClassification(output) => {
            println!("Prediction: class = {}", output.class_name)
        }
        // 他の予測出力タイプもここで処理できます。
    }
}
