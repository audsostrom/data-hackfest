import * as React from 'react';
import Button from '@mui/material/Button';
import CssBaseline from '@mui/material/CssBaseline';
import TextField from '@mui/material/TextField';
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';
import Link from '@mui/material/Link';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import {useEffect} from "react";
import {getServerAuthSession} from "~/server/auth";
import GoogleButton from "~/app/_components/google-button";

export default async function Login() {
  const session = await getServerAuthSession();

  console.log(session);

  return (
    <Container component="main" maxWidth="xs" sx={{
      bgcolor: 'background.paper',
      py: 7,
      px: 4,
    }}>
      <CssBaseline />
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
            gap: '1.25rem',
        }}
      >
        <Typography component="h1" variant="h5">
          Login
        </Typography>
          <Typography
            sx={{
                fontSize: '0.8rem',
                maxWidth: '80%',
                textAlign: 'center',
                margin: 'auto'
            }}
          >
              Start your journey in finding the perfect film to watch!
          </Typography>
          <GoogleButton />
      </Box>
    </Container>
  );
}