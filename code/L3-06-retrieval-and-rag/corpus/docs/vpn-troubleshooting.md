# VPN Troubleshooting Guide

Effective: 1 September 2024. Status: current. Owner: Service desk.

## About the VPN

The company VPN client is called Tunnelgate. It starts automatically when you sign in to a company laptop and connects to the nearest gateway. All traffic goes through the VPN; split tunnelling is not permitted, because it would let unmanaged traffic bypass the web filter.

## Error TG-101: certificate expired

Each laptop holds a device certificate that is valid for one year. When it expires, Tunnelgate shows TG-101 and refuses to connect. To fix it, connect to the office network or a phone hotspot and open the self-service portal, then choose Renew device certificate. The renewal takes about ten minutes.

## Error TG-204: clock out of sync

Tunnelgate rejects the connection if the laptop clock differs from the gateway by more than 5 minutes. This often happens after travel across time zones with automatic time disabled. Turn on automatic date and time, restart the laptop and try again.

## Error TG-309: blocked by a captive portal

Hotel and airport Wi-Fi often requires you to accept terms on a web page before any other traffic is allowed. Tunnelgate cannot connect until that page has been accepted. Disconnect Tunnelgate, open a browser to accept the terms, then reconnect.

## Slow connection

If the VPN connects but is slow, run the speed check in the Tunnelgate menu. Video calls work better when you choose the gateway in your own region. Large file transfers should be scheduled outside peak hours.

## When to call the service desk

If none of the steps above help, call the service desk on extension 4400 and quote the error code shown in the Tunnelgate window.
