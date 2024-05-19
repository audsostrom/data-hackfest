import * as React from 'react';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import {getServerAuthSession} from "~/server/auth";
import GoogleButton from "~/app/_components/google-button";
import {redirect} from "next/navigation";

export default async function Login() {
  const session = await getServerAuthSession();

  if (session) {
      console.log('session', session);
      redirect('/account/' + session.user.id);
  }

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