<style scoped>
  p1{
    margin: 520px 0;
    font:italic 3em Georgia, serif;

  }
  p{
    margin: 12px 0;
    font:italic 2em Georgia, serif;

  }
  
</style>
<template>
  <div >
    <h3><strong1 style="font: italic 2em Georgia, serif;">垃圾邮件检测（中文） </strong1></h3>
      <div >
      </div>
    <a-row>
      <a-col :span="8" :offset="8">
        <a-textarea placeholder="Basic usage" :rows="4" v-model="email"/>
      </a-col>
    </a-row>
    <a-button type="primary" @click="predict">提交</a-button>
    <div><strong1 style="font: italic 2em Georgia, serif;">result: {{result}}</strong1>  </div>
  <div> <h3><strong style>注: ham 是 正常邮件 ; spam 是 骚扰邮件</strong></h3></div>

  
  <br>
 
    <p2>
    <link rel="stylesheet" href="https://unpkg.com/element-ui/lib/theme-chalk/index.css">
      作者：李方正 学号：2019213710 班级：2019215122
      </p2>
      <br>
       <p2> 
         <a href="mailto:lfz@bupt.edu.cn"  strong style= "font:italic 1em Georgia, serif;"> 邮箱：lfz@bupt.edu.cn</a>
        </p2>
        
         <a-collapse v-model: >
      <a-collapse-panel key="1" header="Document_ham">
               <a-card  >
              <p3>你似乎在转换话题
              首先，成绩好，推研地把握就大，这个基本上是废话
              每个人都知道
              但是推研靠的是相对排名而不是绝对成绩，自然会分出名次
              现在我们讨论的重点是现有的推研制度是否有缺陷
              你的观点如下

              “你仍然没有理解我所说的话，学校的制度有问题而导致了混乱
              你应该考虑这个问题：为什么会有学生在那里争夺名额？
              合理的体制的结果是，你三年的学习成绩和其他表现已经决定了你是否会被推研你无需，无意义也不允许去考争夺来获得名额
              你的争夺应该放在前三年的学习成绩上”
              “要做到公平就必须用量化的办法来解决问题，为人处事很重要
              但是这个不该成为决定因素而只是辅助因素，比如同等条件下优先考虑
              否则就会出现学习成绩差靠和老师搞关系而推研的事情
              二者取一，还是学分积能更加客观”
              我的观点很简单，推研这件事情上面要量化非常困难，前面已经说得很明白，从导师的角度出发，选择研究生有很多考虑，不仅仅是看学分积，还要看其他很多因素，例如某些老师就想找一些动手能力强的，某些老师不喜欢学分积高的但是一心要出国的
              就如同招聘一样，要看很多条件，能力强的不一定会适合
                  你能把事情寄托在小概率事件上面吗？
                  成绩好的，肯定可以推研
                  成绩不好的，靠运气也许可以推研
              </p3>
                
            </a-card>
      </a-collapse-panel>
      <a-collapse-panel key="2" header="Document_spam">
        <a-card>
          <p3>大多数人靠打工拿工资，用自己的血汗去成就老板的事业，用自己的辛勤去烘托领导的辉煌。工作40年工资不过千，省吃俭用几十年，买个小套房还要借钱。 
        做生意好赚钱。十多年前第一批被人认为“没出息”的去摆地摊，十多年后的今天个个成了大老板，今天你也要摆地摊，100年也成不了大老板！生意竞争越来越激烈，办工厂？开百货？炒股票？做期货？ 
        有人说，我有技术就好了。其实送你百项专利也知识废纸一张，有几个教授、工程师是富翁？有人说，我要是有资金就好了，其实借你一千万你敢拿过去做保证赚钱的生意？ 有人长期走入赚钱的误区，一提到钱就想到开工厂、做生意。这一想法不突破就抓不住许多在他看来不可能的新机遇。
        真正想一想，成功与失败、富有与贫穷只是因为当初的一念之差。当初要带几千元杀进股市几年后便成了百万富翁。当初只要花几百元愿意去摆地摊10年后就成了大老板。可是有人说，如果我当初做会比他们赚钱更多。不错，是的，你的能力比他强，你的资金比他多，你的经验或许比他足。可是明摆着就是当初一念之差，你的观念决定了你当初不去做，你不去做的观念决定了你10年后的今天还是很穷，不同的观念导致了不同的人生。 
        有人面对一个来之不易的良好机会总是拿不定把握，于是去问他人，问了10人肯定9人说不能做，于是放弃了。其实你不知道机遇来源于新生事物，而新生事物之所以新就是因为90%以上的人还不知道、不认识，等90%的人知道了就不再是新生事物。
        就拿网上赚钱来说，你问10个人，很可能10个人都摇头，但再过一段时间，这10个人点头时，你只能做那10个人的下线了。撑死胆大的饿死胆小的，你是哪一类？难道非要全国人民告诉你网络真的可以赚钱你才加入吗？那个时候什么都晚了! 
        朋友！或许您对网络不太了解和相信，但这毕竟已经成为不争的事实！有很多人收到了人民币汇款！您不妨相信一次，仔细的去研究一下，或许您会有意外的收获,甚至改变你以后的物质生活！ 
        登陆......(不是黄色和有害信息网站，否则你骂我祖宗十八辈！) 在这里白手创业变得不再是难事，我们诚邀您加盟，一起开拓美好未来！这并不影响您干别的事情，但是您却能在畅游网海的同时赚到可观的钱财，还能学习到建网站的知识，一举几得，这样的机会可不多哟。何乐而不为？如果你有兴趣想在上网玩乐的同时赚些钱，不妨停下几分钟看一看。
        希望在这里能找到你所喜爱的东西！也欢迎各位网站的站长与我网站进行友情连接，推广广告。如何赚钱？如何操作？请详细看网站最上面各个栏目！也可以和站长联系或在留言板里留言！

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
      email: '',
      isVisiable:false,
    };
  },
  
  
  
  methods: {
    predict() {
      const path = 'http://localhost:5000/lookUp?email='+this.email;
      console.log(this.email)//测试用的
      axios.post(path)
        .then((res) => {
          this.result = res.data.label;
          console.log(this.result);
          if (this.result=="ham"||this.result=="spam"){
          this.$message.success("查询成功",0.3);
          
          }else {
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
