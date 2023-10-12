<style>
  div{
    margin: 12px 0;
  }
  strong{
    font:italic 1em Georgia, serif;
    color:rgb(145, 143, 143);
  }
</style>


<template>

  <div class="home">
    
    <h3><strong1 style="font: italic 2em Georgia, serif;">Trash Email Division Demo(English) </strong1></h3>
      <!-- <img alt="GitHub stars"
           src="https://img.shields.io/github/stars/zhuzhezhe/mini_sms_classify"
      href="https://github.com/zhuzhezhe/mini_sms_classify"> -->
      
     <a-row>
       
      <a-col :span="8" :offset="8">
        <a-textarea placeholder="Basic usage" :rows="4" v-model="email"/>
      </a-col>
    </a-row>
    <a-button type="primary" @click="predict">submit</a-button>
    <div><strong1 style="font: italic 2em Georgia, serif;">result: {{result}}</strong1>  </div>

    
  <div> <h3><strong style>PS: ham means normal email ; spam means trash email</strong></h3></div>

    <p2>
    <link rel="stylesheet" href="https://unpkg.com/element-ui/lib/theme-chalk/index.css">
      <!-- 作者：李方正 学号：2019213710 班级：2019215122 -->
      Author: MysteriousLi

      </p2>
      <br>
       <p2> 
         <a href="mailto:e1127338@u.nus.edu"  strong style= "font:italic 1em Georgia, serif;"> email： e1127338@u.nus.edu</a>
        </p2>

            <a-collapse v-model: >
      <a-collapse-panel key="1" header="Document_ham">
               <a-card  >
              <p3>I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times.
              </p3>
                
            </a-card>
      </a-collapse-panel>
      <a-collapse-panel key="2" header="Document_spam">
        <a-card>
          <p3>SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info
          </p3>
        </a-card>
      </a-collapse-panel>
         </a-collapse>
  </div>
  
</template>

<script>
    import axios from 'axios'

    export default {
  name: 'home',
  data() {
    return {
      result: '',
      email: ''
    };
  },
  methods: {

    predict() {
      
      const path = 'http://127.0.0.1:5000/predict?email='+this.email;

      // const path = 'http://localhost:5000/predict?email='+this.email;
      console.log(this.email)//测试用的
      axios.post(path)
        .then((res) => {
          this.result = res.data.label;
          console.log(this.result)


      if (res.data.flag=="1")
      {
          this.$message.success("查询成功",0.3);
      } else {
          this.$message.error("查询失败！！！",0.3);
        }


        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
 
    },

  },
  created() {
    // this.predict();
  },



  
}



</script>
