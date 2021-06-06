pipeline {
    agent any
    tools { nodejs "node" }
    stages {
        stage("Git") {
            steps {
                git branch: "master", url: "https://github.com/Losajhonny/IA1_Proyecto.git"
            }
        }
        stage ("install") {
            dir ("App") {
                steps { sh "npm install" }
            }
        }
        stage ("test") {
            steps { sh ".npm test" }
        }
        stage ("deploy") {
            steps { sh "npm start" }
        }
    }
}
