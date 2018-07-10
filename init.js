const EventEmitter = require('events').EventEmitter;
const { spawn } = require('child_process');

const Detector = (() => {

    const _CORE_ENV = "python";
    const _CORE_LIB = "/home/ubuntu/logo-detection/models/research/object_detection/logodetect.py";
    const _CORE_LIB_PARAMS = "";

    const _EVENT_MAP = {
        DATA: 'cnn-response',
        ERR: 'cnn-error',
        END: 'cnn-end'
    };

    function _createProbList(response) {
        try {
            let probList = JSON.parse(response)
            return { probList };
        } catch (err) {
	    //console.log(response);
            return { err };
        }
    }

    return class LogoDetector extends EventEmitter {

        constructor() {
            super();
            this.CORE_ENV = _CORE_ENV;
            this.CORE_LIB = _CORE_LIB;
            this.CORE_LIB_PARAMS = _CORE_LIB_PARAMS;
        }

        coldBoot() {
            try {
                let args = [
                    this.CORE_LIB,
                    ...this.CORE_LIB_PARAMS.split(' ')
                ];

                const spawnedCNN = spawn(this.CORE_ENV, args);

                spawnedCNN.stdout.on('data', (data) => {
              		//console.log(data.toString());
	      	   let { probList, err } = _createProbList(data.toString());
                    if (probList)
                        this.emit(_EVENT_MAP.DATA, probList);
                    else
                        this.emit(_EVENT_MAP.ERR, err);
                });

                spawnedCNN.stderr.on('data', (data) => {
                    this.emit(_EVENT_MAP.ERR, data);
                });

                spawnedCNN.on('close', (code) => {
                    this.emit(_EVENT_MAP.END, { code });
                });
            } catch (err) {
                this.emit(_EVENT_MAP.ERR, err);
            }
        }

    }
})();


module.exports = Detector;

let cnn = new Detector();
cnn.coldBoot();

cnn.on('cnn-response', (data) => {
    console.log(data);
})

cnn.on('cnn-error', (data) => {
    console.log(data);
});

cnn.on('cnn-end', (data) => {
    console.log(data);
});
