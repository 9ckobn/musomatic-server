# 🔒 Security Hardening

Guide for securing your musomatic server when exposed to the internet.

## API Key Authentication

The server supports API key auth for external access. LAN clients are trusted automatically.

```bash
# In .env
API_KEY=your-random-key-here

# Generate a random key:
openssl rand -base64 32
```

**How it works:**
- Requests from `127.*`, `10.*`, `172.16-17.*`, `192.168.*` → trusted (no key needed)
- `/health` endpoint → always public
- Everything else from external IPs → requires `x-api-key` header or `?key=` query param

## Cloudflare Tunnel (Recommended for External Access)

If you're behind CGNAT or can't port-forward, Cloudflare Tunnel is the best option.
It creates an outbound connection from your server to Cloudflare's edge — no inbound ports needed.

```bash
# Install cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared

# Login (opens browser)
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create home-server

# Route DNS
cloudflared tunnel route dns home-server music.yourdomain.com
cloudflared tunnel route dns home-server api.yourdomain.com
```

Create `/root/.cloudflared/config.yml`:
```yaml
tunnel: YOUR_TUNNEL_ID
credentials-file: /root/.cloudflared/YOUR_TUNNEL_ID.json

ingress:
  - hostname: music.yourdomain.com
    service: http://localhost:4533
  - hostname: api.yourdomain.com
    service: http://localhost:8844
  - service: http_status:404
```

Install as service:
```bash
cloudflared service install
systemctl enable cloudflared
systemctl start cloudflared
```

## SSH Hardening

```bash
# /etc/ssh/sshd_config
PermitRootLogin prohibit-password
PasswordAuthentication no
PubkeyAuthentication yes
```

## Firewall (iptables)

Allow only LAN + established connections:

```bash
iptables -I INPUT 1 -i lo -j ACCEPT
iptables -I INPUT 2 -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -I INPUT 3 -s 192.168.0.0/16 -j ACCEPT
iptables -A INPUT -j DROP

# Save rules
mkdir -p /etc/iptables
iptables-save > /etc/iptables/iptables.rules
```

> **Note:** Docker manages its own FORWARD chain rules. The INPUT rules above only affect direct connections to the host — Docker container networking is not affected.

## iOS VPN Compatibility

iOS only allows one VPN connection at a time. If you already use a VPN (WireGuard, VLESS, etc.), you can't add another for musomatic access. That's why Cloudflare Tunnel is ideal — it requires no client-side VPN.
