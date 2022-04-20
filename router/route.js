const express = require('express');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const upload = multer({ dest: "Dataset/" });
const spawn = require('child_process');
const router = express.Router();

function delay(milisecondDelay) {
    milisecondDelay += Date.now();
    while (Date.now() < milisecondDelay) {}
}

function getDateTimeStamp() {
    let arr = (new Date()).toLocaleString('en-GB').split(", ");
    arr[0] = arr[0].split("/").reverse().join("");
    return arr.join("_");
}

router.get('/', (req, res) => {
    res.sendFile(`${__dirname}/public/index.html`);
});

router.post('/uploadCSV', upload.single('ePaymentDataset'), (req, res) => {

    let fileName = req.file.originalname,
        feat_col_arr = [];
    // Rename File
    fs.rename(`${req.file.path}`, `Dataset\\${fileName}`, _ => {});

    // Create Log File
    // Command to Replicate Output
    // let command = `python Python-Executables\\get_features.py '${fileName}'`;
    // let logFileName = `getFeatures_${getDateTimeStamp()}.txt`;
    // fs.writeFile(logFileName, command, err => {
    //     if (err) {
    //         console.log(err);
    //     }
    // });

    let python = spawn.spawnSync('python', ['Python-Executables\\get_features.py', fileName]);

    feat_col_arr = python.stdout.toString().split("->");

    feat_col_arr[feat_col_arr.length - 1] = feat_col_arr[feat_col_arr.length - 1].split("\r\n")[0];

    // python.stdout.on("data", (data) => {
    //     console.log(data.toString().split("->"));
    //     res.json({ feature_columns: feat_col_arr });
    // })

    return res.status(200).json({ msg: `Successfully added ${fileName}!`, feature_columns: feat_col_arr });
});

router.post("/genCorrImage", (req, res) => {
    let filter_feature = req.body.filter_feature,
        target_feature = req.body.target_feature,
        utaut_feature = req.body.utaut_feature,
        fileName = req.body.filename;

    // Convert Array to String
    filter_feature = filter_feature.join("->");
    target_feature = target_feature.join("->");
    utaut_feature = utaut_feature.join("->");

    // Create Log File
    // Command to Replicate Output
    // let command = `python Python-Executables\\gen_corr_img.py '${fileName}' '${utaut_feature}'`;
    // let logFileName = `genCorrImg_${getDateTimeStamp()}.txt`;
    // fs.writeFileSync(`LogFile\\genCorrImg\\${logFileName}`, command);

    let python = spawn.spawnSync('python', ['Python-Executables\\gen_corr_img.py', fileName, utaut_feature]);

    // Run Synchrously
    console.log(python.stdout.toString());

    // command = `python Python-Executables\\gen_features_df.py '${fileName}' '${filter_feature}' '${target_feature}' '${utaut_feature}'`;
    // logFileName = `genFeaturesDf_${getDateTimeStamp()}.txt`;
    // fs.writeFileSync(`LogFile\\genFeaturesDf\\${logFileName}`, command);

    python = spawn.spawnSync('python', ['Python-Executables\\gen_features_df.py', fileName, filter_feature, target_feature, utaut_feature]);

    console.log(python.stdout.toString());

    let rawData = fs.readFileSync("Dataset\\feat_dict.json");
    let feat_dict_obj = JSON.parse(rawData);

    return res.status(200).json({ msg: "Successfully Generated Pearson and Spearman Correlation Image!", feature_dict: feat_dict_obj });
});

router.post("/genViz", (req, res) => {
    let filter_feature = req.body.filter_feature,
        target_feature = req.body.target_feature;

    // Create Log File
    // Command to Replicate Output
    // let command = `python Python-Executables\\gen_viz.py '${filter_feature}' '${target_feature}'`;
    // let logFileName = `genViz_${getDateTimeStamp()}.txt`;
    // fs.writeFileSync(`LogFile\\genViz\\${logFileName}`, command);

    python = spawn.spawnSync('python', ['Python-Executables\\gen_viz.py', filter_feature, target_feature]);

    console.log(python.stdout.toString());
    return res.status(200).json({ msg: "Successfully Generated filter-target-graph.jpeg!" });
});

router.post("/genTrainTest", (req, res) => {
    let filter_feature = req.body.filter_feature,
        target_feature = req.body.target_feature,
        utaut_feature = req.body.utaut_feature;

    // Convert Array to String
    utaut_feature = utaut_feature.join("->");

    // Create Log File
    // Command to Replicate Output
    // let command = `python Python-Executables\\gen_train_test_df.py '${filter_feature}' '${target_feature}' '${utaut_feature}'`;
    // let logFileName = `genTrainTest_${getDateTimeStamp()}.txt`;
    // fs.writeFileSync(`LogFile\\genTrainTest\\${logFileName}`, command);

    // Execute Command
    // python = spawn.spawn('python', ['Python-Executables\\gen_train_test_df.py', filter_feature, target_feature, utaut_feature]);

    // python.stdout.on("data", data => {
    //     console.log(data.toString());
    // })

    // python.stderr.on("data", data => {
    //     console.error(data.toString());
    // });

    python = spawn.spawnSync('python', ['Python-Executables\\gen_train_test_df.py', filter_feature, target_feature, utaut_feature]);

    console.log(python.stdout.toString());

    let rawData;

    rawData = fs.readFileSync("Dataset\\corr_dataFrame.json");
    let corr_dataFrame = JSON.parse(rawData);

    rawData = fs.readFileSync("Dataset\\model_result_dataFrame.json");
    let model_result_dataFrame = JSON.parse(rawData);

    return res.status(200).json({
        msg: "Successfully Selected Important Features using Correlation Feature Selection!",
        cfs_dict: corr_dataFrame,
        model_res_dict: model_result_dataFrame
    });
});

module.exports = router;