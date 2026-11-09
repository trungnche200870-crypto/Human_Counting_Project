'use strict';
const $ = id => document.getElementById(id);
let token = location.hash.slice(1);
if (token) sessionStorage.setItem('cameraToken', token);
token = token || sessionStorage.getItem('cameraToken');
history.replaceState(null, '', location.pathname);
let stream, socket, timer, wakeLock, active = false, pending = false, encoding = false;
let sent = 0, acknowledgedAt = 0, run = 0;
const canvas = document.createElement('canvas');
const context = canvas.getContext('2d');
async function stop(message = 'Đã dừng') {
  active = false; run++; clearInterval(timer); timer = null;
  if (socket) { socket.onclose = null; socket.close(); socket = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  $('video').srcObject = null;
  if (wakeLock) { await wakeLock.release().catch(() => {}); wakeLock = null; }
  pending = encoding = false;
  $('start').disabled = false; $('stop').disabled = true; $('facing').disabled = false;
  $('status').textContent = message;
}
async function capture() {
  if (!active || encoding || !socket || socket.readyState !== WebSocket.OPEN) return;
  if (pending) {
    if (Date.now() - acknowledgedAt > 10000) stop('Mất kết nối. Nhấn Bắt đầu để thử lại.');
    return;
  }
  if (socket.bufferedAmount > 0 || !$('video').videoWidth) return;
  const v = $('video'), scale = Math.min(1, 640 / Math.max(v.videoWidth, v.videoHeight));
  canvas.width = Math.round(v.videoWidth * scale); canvas.height = Math.round(v.videoHeight * scale);
  context.drawImage(v, 0, 0, canvas.width, canvas.height);
  encoding = true; const current = run;
  canvas.toBlob(blob => {
    encoding = false;
    if (!blob || !active || current !== run || socket?.readyState !== WebSocket.OPEN) return;
    socket.send(blob); pending = true; acknowledgedAt = Date.now(); sent += blob.size;
    $('traffic').textContent = `Đã gửi ${(sent / 1048576).toFixed(1)} MB • tối đa 640 px / 10 khung hình mỗi giây`;
  }, 'image/jpeg', .65);
}
$('start').onclick = async () => {
  $('error').textContent = ''; $('start').disabled = true; $('facing').disabled = true;
  try {
    if (!window.isSecureContext) throw new Error('Camera cần đường link HTTPS.');
    if (!token) throw new Error('Thiếu mã ghép nối. Mở link do máy tính cung cấp.');
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Trình duyệt chưa hỗ trợ camera. Hãy dùng Safari.');
    stream = await navigator.mediaDevices.getUserMedia({audio:false, video:{facingMode:{ideal:$('facing').value},width:{ideal:640},height:{ideal:480},frameRate:{ideal:10,max:15}}});
    $('video').srcObject = stream; await $('video').play();
    stream.getVideoTracks()[0].onended = () => stop('Camera đã ngừng. Nhấn Bắt đầu để kết nối lại.');
    socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/camera`);
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Kết nối hết thời gian chờ.')), 15000);
      socket.onopen = () => socket.send(JSON.stringify({token}));
      socket.onmessage = event => {
        const msg = JSON.parse(event.data);
        if (msg.ready) { clearTimeout(timeout); resolve(); }
        if (msg.received) pending = false;
      };
      socket.onerror = () => { clearTimeout(timeout); reject(new Error('Không kết nối được máy tính.')); };
      socket.onclose = event => { clearTimeout(timeout); reject(new Error(event.reason || 'Mã ghép nối sai hoặc camera khác đang kết nối.')); };
    });
    active = true; run++; sent = 0; $('stop').disabled = false;
    socket.onclose = () => stop('Mất kết nối. Nhấn Bắt đầu để kết nối lại.');
    $('status').textContent = 'Đang truyền hình ảnh về máy tính';
    if (navigator.wakeLock) wakeLock = await navigator.wakeLock.request('screen').catch(() => null);
    timer = setInterval(capture, 100);
  } catch (error) { await stop('Chưa kết nối'); $('error').textContent = error.message; }
};
$('stop').onclick = () => stop();
document.addEventListener('visibilitychange', () => { if (document.hidden && stream) stop('Đã dừng khi rời trang. Nhấn Bắt đầu khi quay lại.'); });
window.addEventListener('pagehide', () => stop());
