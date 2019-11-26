require('@tensorflow/tfjs-node');
const plot = require('node-remote-plot');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./logistic-regression');
const loadCSV = require('../load-csv');
const _ = require('lodash');

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv',
	{
		dataColumns: ['horsepower', 'displacement', 'weight'],
		labelColumns: ['mpg'],
		shuffle: true,
		splitTest: 50,
		converters: {
			mpg: value => {
				const mpg = parseFloat(value);
				if (mpg < 15) {
					return [1, 0, 0];
				} else if (mpg < 30) {
					return [0, 1, 0];
				} else {
					return [0, 0, 1];
				}
			}
		}
	});

// console.log(_.flatMap(labels));
const regression = new LogisticRegression(features, _.flatMap(labels), {
	learningRate: 0.5,
	iterations: 100,
	batchSize: 10,
	decisionBoundary: 0.5
});

//
// regression.predict([[130, 307, 1.75]])
//           .print();

regression.train();
console.log(regression.predict([[130, 307, 1.752]]).toString());

console.log(regression.test(testFeatures, _.flatMap(testLabels)));
//
// plot({
//   x: regression.costHistory.reverse()
// });
