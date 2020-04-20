<template>
  <div class="container" @click="clickHandle">
    <div class="message">{{msg}}</div>
  </div>
</template>

<script>
export default {
  data () {
    return {
      msg: 'Hello'
    }
  },

  methods: {
    clickHandle () {
      this.msg = 'Clicked!!!!!!'
    }
  }
}
</script>

<style scoped>
.message {
  color: red;
  padding: 10px;
  text-align: center;
}
</style>