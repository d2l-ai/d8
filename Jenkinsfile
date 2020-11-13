stage("Build and Publish") {
  // such as d2l-en and d2l-zh
  def REPO_NAME = env.JOB_NAME.split('/')[0]
  // such as en and zh
  // def LANG = REPO_NAME.split('-')[1]
  // The current branch or the branch this PR will merge into
  def TARGET_BRANCH = env.CHANGE_TARGET ? env.CHANGE_TARGET : env.BRANCH_NAME
  // such as d2l-en-master
  def TASK = REPO_NAME + '-' + TARGET_BRANCH
  node {
    ws("workspace/${TASK}") {
      checkout scm
      // conda environment
      def ENV_NAME = "${TASK}-${EXECUTOR_NUMBER}";
      // assign two GPUs to each build
      def EID = EXECUTOR_NUMBER.toInteger()
      def CUDA_VISIBLE_DEVICES=(EID*2).toString() + ',' + (EID*2+1).toString();

      sh label: "Build Environment", script: """set -ex
      conda env update -n ${ENV_NAME} -f test.yml
      conda activate ${ENV_NAME}
      pip list
      nvidia-smi
      """

      sh label: "Unit Tests", script: """set -ex
      conda activate ${ENV_NAME}
      d2lbook build outputcheck
      # python -m unittest d8/*.py d8/**/*.py
      # mypy --ignore-missing-imports d8/*.py
      """

      sh label: "Execute Notebooks", script: """set -ex
      conda activate ${ENV_NAME}
      export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
      d2lbook build eval
      """

      sh label:"Build HTML", script:"""set -ex
      conda activate ${ENV_NAME}
      d2lbook build html
      """


      if (env.BRANCH_NAME == 'release') {

      } else {
        sh label:"Publish", script:"""set -ex
        conda activate ${ENV_NAME}
        d2lbook deploy html --s3 s3://preview.d2l.ai/${JOB_NAME}/
        """
        if (env.BRANCH_NAME.startsWith("PR-")) {
            pullRequest.comment("Job ${JOB_NAME}/${BUILD_NUMBER} is complete. \nCheck the results at http://preview.d2l.ai/${JOB_NAME}/")
        }
      }
    }
  }
}
