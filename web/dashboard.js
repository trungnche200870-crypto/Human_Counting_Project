'use strict';
const $ = id => document.getElementById(id);
let token = location.hash.slice(1) || sessionStorage.getItem('adminToken');
history.replaceState(null, '', location.pathname);
let busy = false, url, cameraLink = '', initialized = false;
async function api(path, options = {}) {
  const response = await fetch('/api/' + path, {...options, headers:{'Authorization':'Bearer ' + token, 'Content-Type':'application/json', ...options.headers}});
  if (!response.ok) throw new Error(response.status === 401 ? 'Mã quản trị không đúng hoặc máy chủ đã khởi động lại.' : await response.text());
  return response;
}
function show(state) {
  for (const [key, field] of [['in_count','in'],['out_count','out'],['tracked','tracked'],['fps','fps']]) $(field).textContent = state[key];
  $('status').textContent = state.connected ? 'Camera đang kết nối' : 'Chưa nhận hình / camera đã ngắt — bộ đếm được giữ';
  $('error').textContent = state.error;
  $('direction').textContent = state.reverse ? 'Trên → dưới: OUT | Dưới → trên: IN' : 'Trên → dưới: IN | Dưới → trên: OUT';
  cameraLink = state.camera_link; $('link').textContent = cameraLink || 'Chưa cấu hình địa chỉ HTTPS';
  if (cameraLink) $('link').href = cameraLink; else $('link').removeAttribute('href');
  if (!initialized) {
    $('base').value = state.public_base; $('reverse').checked = state.reverse; $('confidence').value = state.confidence; initialized = true;
  }
}
async function refresh() {
  if (!token || busy) return;
  busy = true;
  try {
    const state = await (await api('status')).json();
    $('login').hidden = true; $('main').hidden = false; sessionStorage.setItem('adminToken',token); show(state);
    const response = await api('frame');
    if (response.status !== 204) {
      const next = URL.createObjectURL(await response.blob()); $('frame').src = next; $('frame').hidden = false;
      if (url) URL.revokeObjectURL(url); url = next;
    } else { $('frame').hidden = true; }
    $('frame').style.opacity = state.connected ? '1' : '.4';
  } catch (error) { $('login').hidden = false; $('status').textContent = 'Không kết nối được máy chủ'; $('error').textContent = error.message; }
  finally { busy = false; }
}
function action(id, handler) { $(id).onclick = async () => { $(id).disabled = true; try { await handler(); } catch (e) { $('error').textContent = e.message; } finally { $(id).disabled = false; } }; }
action('unlock', async () => {token = $('admin').value.trim(); initialized = false; await refresh();});
action('save', async () => show(await (await api('config',{method:'POST',body:JSON.stringify({public_base:$('base').value.trim(),confidence:Number($('confidence').value),reverse:$('reverse').checked})})).json()));
action('reset', async () => {if (confirm('Đặt IN và OUT về 0? Nhật ký cũ vẫn được giữ.')) show(await (await api('reset',{method:'POST'})).json());});
action('disconnect', async () => show(await (await api('disconnect',{method:'POST'})).json()));
action('copy', async () => {if (!cameraLink) throw new Error('Nhập địa chỉ HTTPS và lưu cấu hình trước.'); await navigator.clipboard.writeText(cameraLink);});
action('export', async () => {
  const link = document.createElement('a'), blob = URL.createObjectURL(await (await api('events')).blob());
  link.href = blob; link.download = 'camera_events.csv'; link.click(); setTimeout(() => URL.revokeObjectURL(blob),1000);
});
refresh(); setInterval(refresh, 250);
