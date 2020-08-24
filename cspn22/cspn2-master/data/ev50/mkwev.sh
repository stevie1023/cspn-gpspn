#!/bin/sh

net=$1
cat $net.valid.ev | sort | uniq -c | awk '{print $1"|"$2}' > $net.valid.wev
cat $net.test.ev | sort | uniq -c | awk '{print $1"|"$2}' > $net.test.wev
cat $net.ts.ev | sort | uniq -c | awk '{print $1"|"$2}' > $net.ts.wev
