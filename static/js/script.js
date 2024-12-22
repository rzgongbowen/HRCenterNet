document.querySelector('.img__btn').addEventListener('click', function() {
    document.querySelector('.content').classList.toggle('s--signup')
})

function registerUser() {
    var username = document.getElementById('usernameInput').value;
    var password = document.getElementById('passwordInput').value;
    var confirmPassword = document.getElementById('confirmPasswordInput').value;

    if (!username || !password) {
        document.getElementById('registrationMessage').innerText = '用户名和密码不能为空';
        return;
    }

    if (password !== confirmPassword) {
        document.getElementById('registrationMessage').innerText = '两次密码输入不一致';
        return;
    }

    // 构造要发送的数据
    var data = {
        username: username,
        password: password
    };

    // 发送POST请求给后端
    fetch('/register', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('registrationMessage').innerText = data.error;
        } else {
            document.getElementById('registrationMessage').innerText = '注册成功';
            // 可以根据需要在此处执行其他操作，例如跳转到登录页面
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('registrationMessage').innerText = '注册失败，请稍后重试';
    });
}
